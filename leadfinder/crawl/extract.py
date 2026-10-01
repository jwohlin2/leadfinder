"""Deterministic extraction from fetched HTML/text.

Kept separate from the crawler (which fetches) and the contact resolver (which
grades). Everything here is pure and testable: same input, same output, no
network and no LLM. This is what makes the evidence in a lead reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

# ------------------------------------------------------------------ emails
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
)

_EMAIL_OBFUSCATION = [
    (re.compile(r"\s*\(?at\)?\s*|\s*\[at\]\s*", re.I), "@"),
    (re.compile(r"\s*\(?dot\)?\s*|\s*\[dot\]\s*", re.I), "."),
]

_ROLE_LOCALS = {
    "info", "contact", "hello", "hi", "support", "sales", "admin", "office",
    "enquiries", "enquiry", "inquiries", "inquiry", "mail", "email", "team",
    "help", "helpdesk", "service", "services", "booking", "bookings",
    "reservations", "reservation", "press", "media", "pr", "careers", "jobs",
    "hr", "partnerships", "partners", "bd", "business", "bizdev", "welcome",
    "reception", "front", "desk", "billing", "accounts", "accounting",
    "invoice", "accounts@", "webmaster", "postmaster", "abuse", "noreply",
    "no-reply", "donotreply", "marketing", "newsletter", "legal", "privacy",
    "security", "compliance", "feedback", "customerservice", "customers",
    "stay", "reservations@", "groups", "groups@", "partnership", "collab",
    "collabo", "toi", "owa", "owa@", "all", "everybody", "everyone",
    "contacto", "contactus", "kontakt", "contact@", "vendas", "vertrieb",
    "commercial", "commerciale", "boutique", "ventas", "zakaz", "orders",
    "order", "shop", "shopinfo", "b2b", "wholesale", "export", "import",
    "general", "main", "office@", "hq", "secretary", "vorstand", "marketing@",
}

_PERSONAL_PROVIDERS = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.jp", "yahoo.co.uk",
    "ymail.com", "hotmail.com", "hotmail.co.uk", "outlook.com", "live.com",
    "msn.com", "aol.com", "icloud.com", "me.com", "mac.com", "protonmail.com",
    "proton.me", "pm.me", "gmx.com", "gmx.de", "web.de", "mail.ru", "yandex.ru",
    "qq.com", "163.com", "126.com", "naver.com", "daum.net", "seznam.cz",
    "libero.it", "orange.fr", "free.fr", "wanadoo.fr", "t-online.de",
}

# Senders/asset hosts that appear in HTML but are not company contacts.
_EMAIL_JUNK_SUBSTRINGS = (
    "example.com", "example.org", "domain.com", "yourdomain", "email.com",
    "sentry.io", "wixpress.com", "squarespace.com", "godaddy.com", "cloudflare.com",
    "schema.org", "w3.org", "wordpress.com", "2x.png", ".png", ".jpg", ".jpeg",
    ".gif", ".svg", ".webp", ".css", ".js", "sentry", "no-reply@", "noreply@",
    "donotreply", "postmaster", "webmaster", "abuse@", "mailer-daemon",
    "bounce", "unsubscribe", "@2x", "@3x", "googlemail.com", "reactjs.org",
    "bootstrapcdn", "jsdelivr", "unpkg", "@types", ".png@",
)

_GENERIC_TOKENS = {"email", "e-mail", "mail", "your", "name", "here", "click",
                  "button", "submit", "example", "test", "user", "you"}


def normalize_email(raw: str) -> str:
    """Clean one email candidate. Returns "" when it is not usable."""
    if not raw:
        return ""
    email = raw.strip().strip(".,;:!?\"'<>()[]").lower()
    if ".." in email or email.startswith(".") or email.endswith("."):
        return ""
    local, _, domain = email.partition("@")
    if not local or "." not in domain:
        return ""
    # "www.ireservation.it/en/" + the next table cell collapses into
    # "www.ireserv@ion.it" when whitespace is normalised. A local part that
    # starts with a URL scheme, subdomain or path fragment is that artefact,
    # not an address.
    if local.startswith(("www.", "http", "ftp", "smtp", "mail.")) or "://" in local:
        return ""
    if local.endswith((".", "-", "_", "+")):
        return ""
    if any(token in email for token in _EMAIL_JUNK_SUBSTRINGS):
        return ""
    if local in _GENERIC_TOKENS or len(local) < 2:
        return ""
    if local.replace(".", "").isdigit():
        return ""
    # Strip tracking parameters that HTML mailto links carry.
    return email


def extract_emails(blob: str) -> list[str]:
    """Full email pipeline: deobfuscate, normalise, dedupe, drop junk."""
    out: list[str] = []
    for candidate in deobfuscate_emails(blob):
        value = normalize_email(candidate)
        if value and value not in out:
            out.append(value)
    return out


# Containers that legitimately hold a company's contact details. Mining raw
# HTML instead picks up strings from stylesheets, CDNs and tracking scripts
# (a font URL once yielded "fonts.gst@ic.com" as the best direct email).
_CONTACT_CONTAINER_HINTS = (
    "address", "footer", "contact", "mail", "email", "e-mail", "impressum",
    "legal-notice", "colophon", "about", " Reach out".lower(), "kontakt",
)


def extract_emails_from_html(html: str) -> list[str]:
    """Emails from mailto: links and contact containers only."""
    if not html:
        return []
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out: list[str] = []

    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href.lower().startswith("mailto:"):
            continue
        value = normalize_email(deobfuscate_emails(href[7:].split("?")[0])[0]
                                if deobfuscate_emails(href[7:].split("?")[0]) else "")
        if value and value not in out:
            out.append(value)

    for node in soup.find_all(True):
        if node.name in {"script", "style", "noscript", "template", "svg"}:
            continue
        marker = " ".join(
            str(node.get(attr) or "")
            for attr in ("id", "class", "itemprop", "aria-label", "role")
        ).lower()
        if not any(hint in marker for hint in _CONTACT_CONTAINER_HINTS):
            continue
        for value in extract_emails(node.get_text(" ", strip=True)):
            if value not in out:
                out.append(value)

    return out


def deobfuscate_emails(text: str) -> list[str]:
    """Recover emails written as 'name (at) domain (dot) tld'."""
    out: list[str] = []
    for pattern, replacement in _EMAIL_OBFUSCATION:
        text = pattern.sub(replacement, text or "")
    out.extend(_EMAIL_RE.findall(text))
    return out


def classify_email(email: str, person_names: list[str] | None = None) -> tuple[str, float]:
    """Return (kind, confidence) where kind is 'direct' | 'general' | ''."""
    email = normalize_email(email)
    if not email:
        return "", 0.0
    local, _, domain = email.partition("@")

    for name in person_names or []:
        tokens = [t for t in re.split(r"[\s.]+", normalize_local(name)) if len(t) > 1]
        if not tokens:
            continue
        if local == tokens[0] or local == ".".join(tokens):
            return "direct", 0.8
        if local == tokens[-1] or local == f"{tokens[-1]}.{tokens[0]}":
            return "direct", 0.75
        if local == f"{tokens[0]}{tokens[-1]}" or local == f"{tokens[-1]}{tokens[0]}":
            return "direct", 0.6
        if all(token in local for token in tokens):
            return "direct", 0.55

    if domain in _PERSONAL_PROVIDERS:
        return "direct", 0.5
    if local in _ROLE_LOCALS or any(local.startswith(r) for r in _ROLE_LOCALS if len(r) > 4):
        return "general", 0.7
    if len(local.split(".")[0]) <= 2 and "." in local:
        return "general", 0.6
    # firstname.lastname@company-domain style on the official domain
    if "." in local:
        return "direct", 0.45
    return "general", 0.4


def normalize_local(name: str) -> str:
    text = re.sub(r"[^\w\s]", " ", (name or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def extract_phones_from_html(html: str) -> list[str]:
    """Phones from contact containers, which main-content extraction drops.

    Footers and address blocks are exactly where a small company publishes its
    number, and trafilatura usually discards them.
    """
    if not html:
        return []
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out: list[str] = []
    for node in soup.find_all(True):
        if node.name in {"script", "style", "noscript", "template", "svg"}:
            continue
        marker = " ".join(
            str(node.get(attr) or "")
            for attr in ("id", "class", "itemprop", "aria-label")
        ).lower()
        if not any(hint in marker for hint in _CONTACT_CONTAINER_HINTS):
            continue
        for anchor in node.find_all("a", href=True):
            href = str(anchor["href"]).strip()
            if href.lower().startswith("tel:"):
                value = re.sub(r"[^\d+]", " ", href[4:]).strip()
                digits = re.sub(r"\D", "", value)
                if 7 <= len(digits) <= 15 and value not in out:
                    out.append(value)
        for value in extract_phones(node.get_text(" ", strip=True)):
            if value not in out:
                out.append(value)
    return out[:8]


# ------------------------------------------------------------------ phones
_PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+|00)?\d[\d\s().\-]{6,18}\d(?![\w.])"
)
_PHONE_JUNK = re.compile(r"(\d{4,}\s*\d{4,})")
_PHONE_LABEL = re.compile(
    r"(?:tel|phone|fax|mobile|call|whatsapp|telefon|tel\u00e9fono|telefono|"
    r"telf|contatto|mobi|hotline|support)", re.I
)


def extract_phones(text: str) -> list[str]:
    """Pull plausible phone numbers, preferring ones near a phone label."""
    if not text:
        return []
    candidates: list[tuple[int, str]] = []
    for line in (text or "").splitlines():
        labelled = bool(_PHONE_LABEL.search(line))
        for match in _PHONE_RE.finditer(line):
            raw = match.group(0).strip()
            digits = re.sub(r"\D", "", raw)
            if not 7 <= len(digits) <= 15:
                continue
            # An unbroken run of >11 digits is a tracking id, not a number.
            # Anything with separators is a formatted phone number.
            separated = bool(re.search(r"[\s().\-+]", raw))
            if not separated and len(digits) > 11:
                continue
            if re.fullmatch(r"20\d{6,}", digits):  # bare year/registration noise
                continue
            if re.fullmatch(r"(\d)\1+", digits):
                continue
            candidates.append((0 if labelled else 2, raw))
    seen: set[str] = set()
    out: list[str] = []
    for _, raw in sorted(candidates, key=lambda item: item[0]):
        normalized = re.sub(r"\s+", " ", raw).strip(" .-")
        digits = re.sub(r"\D", "", normalized)
        if digits in seen or len(digits) < 7:
            continue
        seen.add(digits)
        out.append(normalized)
    return out[:8]


# ----------------------------------------------------------------- socials
_SOCIAL_HOSTS = {
    "linkedin.com": ("linkedin", "https://linkedin.com"),
    "twitter.com": ("twitter", "https://twitter.com"),
    "x.com": ("twitter", "https://x.com"),
    "facebook.com": ("facebook", "https://facebook.com"),
    "instagram.com": ("instagram", "https://instagram.com"),
    "youtube.com": ("youtube", "https://youtube.com"),
    "github.com": ("github", "https://github.com"),
    "gitlab.com": ("gitlab", "https://gitlab.com"),
    "medium.com": ("medium", "https://medium.com"),
    "crunchbase.com": ("crunchbase", "https://crunchbase.com"),
    "producthunt.com": ("producthunt", "https://producthunt.com"),
    "xing.com": ("xing", "https://xing.com"),
    "weibo.com": ("weibo", "https://weibo.com"),
    "note.com": ("note", "https://note.com"),
    "gitee.com": ("gitee", "https://gitee.com"),
    "t.me": ("telegram", "https://t.me"),
    "wa.me": ("whatsapp", "https://wa.me"),
}


def extract_socials(urls: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for url in urls:
        host = (urlparse(url).hostname or "").lower().removeprefix("www.")
        for needle, (key, base) in _SOCIAL_HOSTS.items():
            if host == needle or host.endswith("." + needle):
                if key not in found:
                    found[key] = url
                if key == "linkedin" and "/in/" in url:
                    found["linkedin_profile"] = url
                break
    return found


_WECHAT_RE = re.compile(
    r"(?:wechat|weixin|微信|wechat\s*id|weixin\s*id|wx\s*id)\s*[:：=]?\s*"
    r"([A-Za-z][-_A-Za-z0-9]{5,19})", re.I
)


def extract_wechat(text: str) -> str:
    match = _WECHAT_RE.search(text or "")
    return match.group(1) if match else ""


# ----------------------------------------------------------- contact forms
_CONTACT_PATH_HINTS = (
    "contact", "kontakt", "contacto", "contatti", "contato", "contactez",
    "get-in-touch", "getintouch", "reach", "inquiry", "enquiry", "support",
    "help", "impressum", "about", "hello", "talk", "book", "quote",
    "reserve", "reservation", "demo", "callback", "sales", "iletisim",
    "kontakta", "napisz", "pisz", "kontaktuj", "porcontacto", "fale-conosco",
    "お問い合わせ", "次序", "聯絡", "联系",
)
_FORM_HINTS = ("form", "contact", "inquiry", "newsletter", "subscribe")

# Forms that exist on nearly every site and are never a contact path.
_NON_CONTACT_FORM_HINTS = (
    "cookie", "consent", "gdpr", "ccookie", "cc-consent", "preference",
    "newsletter", "subscribe", "subscription", "unsubscribe", "mailing",
    "search", "login", "signin", "sign-in", "log-in", "register", "signup",
    "sign-up", "account", "password", "recaptcha", "captcha", "comment",
    "reply", "track", "pixel", "survey", "vote", "rating", "review",
    "waitlist", "invite", "referral", "affiliate", "collab", "partnership",
    "demo-request", "unsubscribe", "logout", "password-reset", "otp",
)

# A real contact form needs a way to say something *and* a way to say who you
# are. Requiring both is what separates it from a search box.
_CONTACT_FIELD_TOKENS = (
    "email", "e-mail", "mail", "name=", "your name", "full name", "first name",
    "phone", "tel", "telefon", "company", "organisation", "organization",
    "subject", "betreff", "asunto", "oggetto",
)
_MESSAGE_FIELD_TOKENS = (
    "message", "textarea", "nachricht", "messaggio", "mensaje", "comment",
    "your message", "how can we help", "tell us", "description", "frage",
    "お問い合わせ内容", "お問い合わせ", "留言", "咨询",
)


def is_contact_page(url: str, title: str = "", text: str = "") -> bool:
    blob = f"{url} {title}".lower()
    if any(hint in blob for hint in _CONTACT_PATH_HINTS):
        return True
    if any(hint in title.lower() for hint in _FORM_HINTS):
        return True
    return False


def find_contact_forms(html: str, page_url: str) -> list[str]:
    """Return URLs of pages that expose a usable contact form.

    Cookie-consent, newsletter and search widgets are excluded, and a form only
    counts when it can receive both an identity and a message - otherwise every
    site with a newsletter would look contactable.
    """
    from bs4 import BeautifulSoup

    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    found: list[str] = []
    for form in soup.find_all("form"):
        haystack = " ".join(
            filter(
                None,
                [
                    str(form.get("action") or ""),
                    str(form.get("id") or ""),
                    str(form.get("class") or ""),
                    str(form.get("name") or ""),
                    str(form.get("aria-label") or ""),
                    form.get_text(" ", strip=True)[:600],
                ],
            )
        ).lower()
        if any(hint in haystack for hint in _NON_CONTACT_FORM_HINTS):
            continue

        # Inspect the actual controls rather than trusting surrounding copy.
        control_blob = " ".join(
            str(element.get(attr) or "")
            for element in form.find_all(["input", "textarea", "select", "button"])
            for attr in ("name", "id", "class", "placeholder", "aria-label", "type")
        ).lower()
        blob = f"{haystack} {control_blob}"

        has_message = any(token in blob for token in _MESSAGE_FIELD_TOKENS)
        has_identity = any(token in blob for token in _CONTACT_FIELD_TOKENS)
        names_a_contact_form = any(
            hint in haystack for hint in ("contact", "inquiry", "enquiry", "reach", "kontakt")
        )
        if not ((has_message and has_identity) or (names_a_contact_form and has_identity)):
            continue

        action = form.get("action")
        if action and not str(action).lower().startswith("javascript:"):
            target = urljoin(page_url, str(action))
            parsed = urlparse(target)
            if parsed.scheme in {"http", "https"} and not _is_search_endpoint(target):
                found.append(target)
        elif page_url not in found:
            found.append(page_url)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for url in found:
        key = url.rstrip("/")
        if key not in seen:
            seen.add(key)
            out.append(url)
    return out


_SEARCH_ENDPOINTS = ("google.", "bing.com/search", "duckduckgo.com", "search?")


def _is_search_endpoint(url: str) -> bool:
    lowered = url.lower()
    return any(token in lowered for token in _SEARCH_ENDPOINTS)


# -------------------------------------------------------------- legal names
_LEGAL_SUFFIXES = (
    "ltd", "ltd.", "limited", "llc", "l.l.c.", "inc", "inc.", "incorporated",
    "corp", "corp.", "corporation", "gmbh", "mbh", "ag", "kg", "ohg", "gbr",
    "co", "co.", "company", "plc", "s.a.", "sa", "sas", "sarl", "s.l.", "sl",
    "s.r.l.", "srl", "b.v.", "bv", "nv", "n.v.", "pty", "pty ltd", "pvt",
    "private limited", "pte", "pte ltd", "ab", "oy", "as", "aps", "a/s",
    "sp. z o.o.", "sro", "s.r.o.", "doo", "dooel", "ood", "eood", "kft",
    "sarl", "sas", "sl", "as", "oyj", "aoy", "pte ltd",
    "sdn bhd", "bhd", "pt", "ltda", "cv", "bvba", "cvba", "ood",
    "aps", "as", "kft", "z.o.o.", "sp.z.o.o.", "s.r.l.",
)
_JP_LEGAL = re.compile(
    r"(?:[\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]{2,20}株式会社|"
    r"株式会社[\u30A0-\u30FF\u4E00-\u9FFF]{2,20})"
)
_CN_LEGAL = re.compile(r"[\u4E00-\u9FFF]{2,24}?(?:股份有限公司|有限责任公司|有限公司)")
_LATIN_LEGAL = re.compile(
    r"\b(?:[A-Z][\w&'’\-\.]*\s+){0,5}[A-Z][\w&'’\-\.]*\s+"
    r"(?:Ltd\.?|Limited|LLC|Inc\.?|Incorporated|Corp\.?|Corporation|GmbH|mbH|"
    r"AG|SA|S\.A\.|SAS|SARL|SL|S\.L\.|B\.V\.|N\.V\.|Pty(?:\s+Ltd)?|Pte(?:\s+Ltd)?|"
    r"AB|Oy|APS|S\.p\.?\s*z\s*o\.o\.|s\.r\.o\.|srl|S\.r\.l\.|s\.a\.r\.l\.|"
    r"d\.o\.o\.|LLC)"
    # A legal suffix usually ends in '.', and a trailing '.' is followed by
    # punctuation, not a word - so \b would never match here. Assert "not a
    # word character" instead.
    r"(?![A-Za-z0-9])"
)
_COPYRIGHT_ENTITY = re.compile(
    r"(?:©|&copy;|copyright)\s*[^0-9\n]{0,10}?((?:[A-Z][\w&'’\-\.]*\s+){1,6}"
    r"[A-Z][\w&'’\-\.]*)", re.I
)


def extract_legal_names(text: str) -> list[str]:
    """Legal entity names in Latin, Japanese and Chinese forms."""
    if not text:
        return []
    found: list[str] = []

    for match in _JP_LEGAL.findall(text):
        value = match.strip()
        if value and value not in found:
            found.append(value)
    for match in _CN_LEGAL.findall(text):
        value = match.strip()
        if value and value not in found:
            found.append(value)
    for match in _LATIN_LEGAL.findall(text):
        value = re.sub(r"\s+", " ", match).strip(" ,.;")
        if not (2 <= len(value) <= 70):
            continue
        tokens = value.split()
        if len(tokens) < 2:
            continue
        if all(token.lower() in _GENERIC_TOKENS for token in tokens):
            continue
        if value not in found:
            found.append(value)

    # Trim the Latin matches down to the tightest legal-name span so that
    # "Availroom is a product of TravelBy Software S.L." yields
    # "TravelBy Software S.L." rather than the whole sentence.
    return _tighten_legal_names(found)


def _tighten_legal_names(names: list[str]) -> list[str]:
    tight: list[str] = []
    for name in names:
        tokens = name.split()
        start = 0
        for index, token in enumerate(tokens):
            if token.lower().strip(".,") in _GENERIC_TOKENS:
                start = index + 1
        trimmed = " ".join(tokens[start:]).strip(" ,.;")
        if len(trimmed.split()) >= 2 and trimmed not in tight:
            tight.append(trimmed)
        elif trimmed and trimmed not in tight and name not in tight:
            tight.append(trimmed)
    return tight


def extract_copyright_entity(text: str) -> str:
    match = _COPYRIGHT_ENTITY.search(text or "")
    if not match:
        return ""
    entity = re.sub(r"\s+", " ", match.group(1)).strip(" .,;")
    if len(entity.split()) > 6 or len(entity) > 60:
        return ""
    return entity


# ---------------------------------------------------------- registry codes
_REGISTRY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("jp_corporate_number", re.compile(r"(?:法人番号|法人No\.?)\s*[:：]?\s*(\d{13})")),
    ("cn_usci", re.compile(r"统一社会信用代码\s*[:：]?\s*([0-9A-HJ-NPQRTUWXY]{18})")),
    ("de_register", re.compile(r"(?:HRB|HR|Register(?:gericht)?)\s*[:：]?\s*(\d{3,7})\b")),
    ("uk_company_number", re.compile(
        r"(?:Company\s*(?:number|no\.?)|Registered\s+(?:in\s+England|number))\s*"
        r"[:：]?\s*(?:No\.?\s*)?(\d{6,8})\b", re.I)),
    ("es_cif", re.compile(r"\b([A-Z]?\d{7,8}[A-Z]?)\b")),
    ("it_rea", re.compile(r"(?:REA|Codice\s+Fiscale)\s*[:：]?\s*([A-Z0-9]{9,16})\b", re.I)),
    ("au_abn", re.compile(r"\bABN\s*[:：]?\s*(\d{2}\s?\d{3}\s?\d{3}\s?\d{3})\b", re.I)),
    ("us_ein", re.compile(r"\b(?:EIN|Employer\s+Identification)\s*[:：]?\s*(\d{2}-\d{7})\b", re.I)),
    ("fr_siren", re.compile(r"\b(?:SIREN|SIRET)\s*[:：]?\s*(\d{9})\b", re.I)),
    ("nl_kvk", re.compile(r"\b(?:KvK|Kamer\s*van\s*Koophandel)\s*[:：]?\s*(\d{8})\b", re.I)),
]


def extract_registry_codes(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for kind, pattern in _REGISTRY_PATTERNS:
        for match in pattern.finditer(text or ""):
            value = match.group(1).strip()
            if (kind, value) not in out:
                out.append((kind, value))
            if len(out) >= 8:
                return out
    return out


# ------------------------------------------------------------------ people
_NAME_TOKEN = r"[A-Z][\w'’\-\.]+"
_CJK = r"[\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]"

_TITLE_LIST = (
    "founder", "co-founder", "cofounder", "owner", "chief executive officer",
    "ceo", "chief operating officer", "coo", "chief technology officer", "cto",
    "chief financial officer", "cfo", "chief information officer", "cio",
    "chief marketing officer", "cmo", "president", "vice president", "vp",
    "managing director", "managing partner", "general manager", "head of",
    "director", "partner", "principal", "manager", "lead", "engineer",
    "developer", "designer", "consultant", "coordinator", "specialist",
    "board member", "advisor", "supervisor", "chairman", "chairwoman",
    "representative director", "executive officer", "senior advisor",
    "product manager", "product owner", "sales director", "account executive",
    "business development", "operations manager", "marketing director",
    # Department words that start compound titles ("Sales & Marketing
    # Manager", "Marketing & Communications Director"). The title must still
    # be short and free of prose/chrome, so these stay role phrases rather
    # than free-text openers.
    "sales", "marketing", "operations", "engineering", "finance",
    "human resources", "customer success", "communications", "growth",
    "content", "creative",
    # App-store listings brand the *app* as "App Name - Developer Name", so
    # "developer" alone matches store chrome rather than an employment title.
    "app store", "google play", "the developer", "developer of",
)
_CJK_TITLES = (
    "代表取締役", "代表取締役社長", "代表取締役ceo", "会長", "社長", "代表", "代表者",
    "創立者", "創業者", "共同創業者", "創業者", "法人代表", "代表取締役",
    "創業者", "創業者", "cto", "cieo", "，愿", "负责人", "创始人", "法定代表人",
    "董事長", "董事長兼ceo", "執行長", "總經理", "總裁", "執行長",
    "viceleiter", "geschäftsführer", "inhaber", "vertreter", "ansprechpartner",
)
_TITLE_ALTERNATION = "|".join(
    re.escape(t) for t in sorted(set(_TITLE_LIST) | set(_CJK_TITLES), key=len, reverse=True)
)
_TITLE_RE = re.compile(rf"(?:{_TITLE_ALTERNATION})", re.I)
# A title must *begin* with a recognised role word. Requiring the title to start
# with one is what stops "Channel Manager - <marketing sentence>" from being
# read as a person named "Channel Manager".
_TITLE_STARTS_RE = re.compile(rf"^\s*(?:{_TITLE_ALTERNATION})\b", re.I)
_MAX_TITLE_LEN = 60
# Sentence punctuation means we grabbed prose, not a job title.
_TITLE_PROSE_RE = re.compile(r"[.!?;,]|(?:\s(?:and|the|for|with|und|et|y|e|per)\s)")
# "Founder & CEO - Paris" style suffixes are fine, but a long dash-separated
# run of page chrome ("App Store - Apple - The developer") is not a title.
_TITLE_CHROME_RE = re.compile(r"\s[-–—|]\s.*[-–—|]\s")

# Tokens that mark a capitalised phrase as a product or department name rather
# than a human being. Needed because "Property Management System" and "Channel
# Manager" are indistinguishable from a name by capitalisation alone.
_PRODUCT_TOKENS = {
    "management", "system", "systems", "property", "properties", "channel",
    "channels", "product", "products", "software", "solution", "solutions",
    "platform", "platforms", "service", "services", "suite", "engine",
    "booking", "bookings", "reservation", "reservations", "manager", "master",
    "cloud", "desk", "portal", "panel", "dashboard", "api", "app", "apps",
    "module", "modules", "edition", "standard", "premium", "basic", "pro",
    "one", "plus", "lite", "max", "hub", "center", "centre", "office",
    "operations", "operations", "revenue", "yield", "rate", "rates",
    "inventory", "rates", "pms", "vrms", "crm", "erp", "cms", "crm",
}

# Words that mean the candidate is a role, not a name.
_ROLE_TOKENS = {
    "founder", "cofounder", "owner", "ceo", "cfo", "coo", "cto", "cio", "cmo",
    "president", "vice", "head", "chief", "director", "manager", "lead",
    "partner", "principal", "engineer", "developer", "designer", "consultant",
    "coordinator", "specialist", "advisor", "supervisor", "chairman",
    "chairwoman", "representative", "executive", "officer", "team", "staff",
}

_CANDIDATE_NAME_RE = re.compile(
    rf"(?<![\w@.])(?:{_NAME_TOKEN}(?:[\s\-.’']+{_NAME_TOKEN}){{0,3}}|{_CJK}{{2,4}})(?![\w@.])"
)
_SENTENCE_SPLIT = re.compile(r"[.!?\n\r•·|]+")

_TITLE_TO_NAME = re.compile(
    rf"^\s*(?P<title>(?:{_TITLE_ALTERNATION})[\w\s/&.\-]{{0,40}}?)\s*[:：\-–—]\s*"
    rf"(?P<name>(?:{_NAME_TOKEN}(?:[\s\-.’']+{_NAME_TOKEN}){{0,3}}|{_CJK}{{2,4}}))\s*$",
    re.I,
)
_NAME_TO_TITLE = re.compile(
    rf"^\s*(?P<name>(?:{_NAME_TOKEN}(?:[\s\-.’']+{_NAME_TOKEN}){{0,3}}|{_CJK}{{2,4}}))"
    rf"\s*[,，–—\-:|]\s*(?P<title>(?:{_TITLE_ALTERNATION})[\w\s/&.\-]{{0,40}}?)\s*$",
    re.I,
)

# Company legal suffixes ("Duetto Co", "Acme Inc.") and generic non-name
# words. "Co" has no business in a human name, and neither do tech-stack
# labels ("Google Fonts") or table chrome ("Other Performer").
_COMPANY_SUFFIX_TOKENS = {
    "co", "cos", "corp", "corps", "inc", "ins", "ltd", "llc", "gmbh",
    "sarl", "srl", "bv", "bvba", "nv", "ag", "oü", "oy", "ab", "spa",
    "plc", "llp", "kft", "doo", "d.o.o.", "pte", "pty", "sa", "s.a.",
}
_TECH_STACK_TOKENS = {
    "fonts", "tag", "tags", "performer", "other", "year", "album", "songs",
    "song", "records", "analysis", "analytics", "stack", "webflow",
    "cloudflare", "google", "open", "graph", "optimization", "version",
    "hosting", "database", "url", "http", "https", "ssl", "cdn", "api",
    "widget", "widgets", "plugin", "plugin", "font", "track", "tracks",
    "discography", "genre", "label", "titled", "single", "singles",
}
_NON_PERSON_TOKENS = {
    "privacy", "cookie", "terms", "contact", "about", "home", "search", "login",
    "sign", "our", "the", "and", "for", "with", "from", "your", "team", "blog",
    "news", "press", "careers", "jobs", "help", "faq", "support", "services",
    "products", "pricing", "features", "legal", "impressum", "sitemap",
    "copyright", "all", "rights", "reserved",
    "company", "corporation", "group", "international", "solutions", "systems",
    "technologies", "labs", "studio", "agency", "partners", "capital", "holdings",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "menu", "close", "read", "more", "view", "learn", "get", "start",
    "free", "trial", "demo", "book", "now", "today", "explore", "discover",
    *_COMPANY_SUFFIX_TOKENS, *_TECH_STACK_TOKENS,
}


def _looks_like_person_name(candidate: str) -> bool:
    value = candidate.strip()
    if not value or len(value) < 3 or len(value) > 60:
        return False
    if re.search(r"@|https?://|\d{3}", value):
        return False
    if re.search(rf"^{_CJK}+$", value):
        # CJK: accept 2-4 characters, reject pure punctuation/digits.
        return 2 <= len(value) <= 4
    tokens = [t for t in re.split(r"[\s\-.’']+", value) if t]
    # A single capitalised word is far more often a product, department or menu
    # label than a person. Require at least two tokens for Latin-script names.
    if not 2 <= len(tokens) <= 4:
        return False
    lowered = [t.lower().strip(".,") for t in tokens]
    if any(token in _NON_PERSON_TOKENS for token in lowered):
        return False
    # A person's name does not contain job titles or product vocabulary.
    if any(token in _ROLE_TOKENS or token in _PRODUCT_TOKENS for token in lowered):
        return False
    if not all(token[:1].isupper() for token in tokens):
        return False
    if any(len(token) > 18 for token in tokens):
        return False
    return True


def _looks_like_title(candidate: str) -> bool:
    """A job title starts with a role word and is short, not a sentence."""
    value = (candidate or "").strip()
    if not value or len(value) > _MAX_TITLE_LEN:
        return False
    if not _TITLE_STARTS_RE.match(value):
        return False
    if _TITLE_PROSE_RE.search(value):
        return False
    if _TITLE_CHROME_RE.search(value):
        return False
    return True


@dataclass(slots=True)
class PersonMention:
    name: str
    title: str = ""
    context: str = ""

    def as_tuple(self) -> tuple[str, str, str]:
        return self.name, self.title, self.context


def extract_people(html: str, text: str) -> list[PersonMention]:
    """Find (name, title) pairs from tables, text lines and markup adjacency."""
    mentions: list[PersonMention] = []
    seen: set[tuple[str, str]] = set()

    def add(name: str, title: str, context: str) -> None:
        name = re.sub(r"\s+", " ", name).strip(" ,.;:|·-–—")
        title = re.sub(r"\s+", " ", title).strip(" ,.;:|·-–—")[:_MAX_TITLE_LEN]
        if not name or not _looks_like_person_name(name):
            return
        # A name with no credible title attached is not a person finding; it
        # is almost always a heading or a product label.
        if not _looks_like_title(title):
            return
        key = (name.lower(), title.lower())
        if key in seen:
            return
        seen.add(key)
        mentions.append(PersonMention(name=name, title=title, context=context[:400]))

    # 1. Team tables: <tr><td>Jane Smith</td><td>CEO</td></tr> is the most
    #    reliable machine-readable shape on a company site.
    if html:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        for row in soup.find_all("tr"):
            cells = [
                cell.get_text(" ", strip=True)
                for cell in row.find_all(["td", "th"])
            ]
            if len(cells) < 2:
                continue
            for index, cell in enumerate(cells[:-1]):
                if not _looks_like_person_name(cell):
                    continue
                following = cells[index + 1]
                if _looks_like_title(following):
                    add(cell, following, " | ".join(cells))
                    break
        # 2. Heading followed by a name or role paragraph.
        for node in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
            heading = node.get_text(" ", strip=True)
            if not _looks_like_person_name(heading):
                continue
            for sibling in list(node.next_siblings)[:3]:
                sibling_text = (
                    sibling.get_text(" ", strip=True)
                    if hasattr(sibling, "get_text")
                    else str(sibling).strip()
                )
                if _looks_like_title(sibling_text):
                    add(heading, sibling_text, f"{heading} — {sibling_text}")
                    break
        # 3. Inline pairs in a single element: <p>Jane Smith, CEO</p> or
        #    <li>Founder - Jane Smith</li>. Very common on team and about
        #    pages, and invisible to the main-content text extractor.
        for node in soup.find_all(["p", "li", "span", "div", "strong", "em", "b"]):
            if node.find(["p", "li", "table", "address", "footer"]):
                continue
            line = node.get_text(" ", strip=True)
            if not line or len(line) > 140:
                continue
            match = _NAME_TO_TITLE.match(line)
            if match:
                add(match.group("name"), match.group("title"), line)
                continue
            match = _TITLE_TO_NAME.match(line)
            if match:
                add(match.group("name"), match.group("title"), line)

    # 3. Text lines: "Jane Smith, Founder" / "Founder - Jane Smith".
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = _TITLE_TO_NAME.match(line)
        if match and _looks_like_person_name(match.group("name")):
            add(match.group("name"), match.group("title"), line)
            continue
        match = _NAME_TO_TITLE.match(line)
        if match and _looks_like_person_name(match.group("name")):
            add(match.group("name"), match.group("title"), line)
            continue
        # Inline in a sentence: "... Jane Smith, CEO and ...". Only short
        # segments qualify - a long line is prose, not a name/title pair.
        if len(line) > 140 or not _TITLE_RE.search(line):
            continue
        for sentence in _SENTENCE_SPLIT.split(line):
            for piece in re.split(r"\s+(?:and|and\s+the|、|と)\s+", sentence):
                match = _NAME_TO_TITLE.match(piece.strip())
                if match and _looks_like_person_name(match.group("name")):
                    add(match.group("name"), match.group("title"), sentence)
                    break
            else:
                continue
            break

    # 4. Adjacent text lines: a name line directly above a title line. Team
    #    pages built from div grids (no table or heading markup) extract as
    #    "Frank Verhagen\nFounder & President", which none of the same-line
    #    rules can see.
    lines = [line.strip() for line in (text or "").splitlines()]
    for index in range(len(lines) - 1):
        first, second = lines[index], lines[index + 1]
        if not first or not second:
            continue
        if _looks_like_person_name(first) and _looks_like_title(second):
            add(first, second, f"{first} — {second}")
        elif _looks_like_title(first) and _looks_like_person_name(second):
            add(second, first, f"{first} — {second}")

    return mentions


# --------------------------------------------------------------- addresses
_ADDRESS_HINTS = (
    "address", "head office", "headquarter", "hq", "office", "located in",
    "based in", "所在地", "住所", "本社", "所在地", "地址", "总部", "所在地",
)
_ADDRESS_LINE = re.compile(
    r"\b\d{1,5}\s+[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*){0,4}\s*,?\s*"
    r"(?:[A-Z][\w'’\-]*\s*){0,3},?\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"
)
_CJK_ADDRESS = re.compile(
    r"[\u4E00-\u9FFF]{2,10}(?:市|区|町|村)[\u4E00-\u9FFF0-9\-]{2,20}"
    r"(?:\d+[-ー丁目\-番地]?\d*)"
)
_JP_ADDRESS = re.compile(
    r"[\u4E00-\u9FFF]{2,8}?[都道府県][\u4E00-\u9FFF]{2,8}?[市区町村]"
    r"[\u4E00-\u9FFF\d\-]{2,20}"
)


def extract_addresses(text: str) -> list[str]:
    out: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or len(line) > 200:
            continue
        if not any(hint in line.lower() for hint in _ADDRESS_HINTS):
            continue
        for pattern in (_ADDRESS_LINE, _JP_ADDRESS, _CJK_ADDRESS):
            match = pattern.search(line)
            if match:
                value = match.group(0).strip(" ,;")
                if value not in out:
                    out.append(value)
                break
    if not out:
        for pattern in (_JP_ADDRESS, _CJK_ADDRESS, _ADDRESS_LINE):
            for match in pattern.finditer(text or ""):
                value = match.group(0).strip(" ,;")
                if value not in out:
                    out.append(value)
                if len(out) >= 3:
                    break
            if out:
                break
    return out[:5]


# ------------------------------------------------------------------ bundle
@dataclass(slots=True)
class PageFacts:
    emails: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    socials: dict[str, str] = field(default_factory=dict)
    wechat: str = ""
    legal_names: list[str] = field(default_factory=list)
    copyright_entity: str = ""
    registry_codes: list[tuple[str, str]] = field(default_factory=list)
    people: list[PersonMention] = field(default_factory=list)
    addresses: list[str] = field(default_factory=list)
    contact_forms: list[str] = field(default_factory=list)


def extract_facts(url: str, html: str, text: str) -> PageFacts:
    facts = PageFacts()
    for email in extract_emails(text or ""):
        if email not in facts.emails:
            facts.emails.append(email)
    for email in extract_emails_from_html(html or ""):
        if email not in facts.emails:
            facts.emails.append(email)
    facts.phones = extract_phones(text or "")
    for phone in extract_phones_from_html(html or ""):
        if phone not in facts.phones:
            facts.phones.append(phone)
    facts.phones = facts.phones[:8]
    from bs4 import BeautifulSoup

    hrefs: list[str] = []
    if html:
        soup = BeautifulSoup(html, "lxml")
        hrefs = [
            urljoin(url, anchor.get("href", ""))
            for anchor in soup.find_all("a", href=True)
        ]
        facts.contact_forms = find_contact_forms(html, url)
    facts.socials = extract_socials(hrefs)
    facts.wechat = extract_wechat(text or "")
    facts.legal_names = extract_legal_names(f"{text or ''}\n{' '.join(hrefs)}")
    facts.copyright_entity = extract_copyright_entity(text or "")
    facts.registry_codes = extract_registry_codes(text or "")
    facts.people = extract_people(html or "", text or "")
    facts.addresses = extract_addresses(text or "")
    return facts
