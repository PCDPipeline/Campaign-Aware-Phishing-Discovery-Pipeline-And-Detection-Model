"""
Helper functions used across the phishing campaign discovery pipeline.

This module provides utilities for:
• URL and path extraction
• FQDN and domain feature extraction
• Statistical summaries of domain patterns
• Entropy and structural pattern analysis

These helpers support feature engineering and rule generation stages.
"""

from urllib.parse import urlparse
from collections import Counter
from math import log2
import re
import tldextract
from tldextract import extract


# ---------------------------------------------------------------------------
# URL / PATH UTILITIES
# ---------------------------------------------------------------------------

def extract_partial_path(url: str) -> str:
    """
    Extract the last 3 path segments of a URL (excluding the file name).

    Used to create stable path identifiers for clustering when full paths
    are overly specific or contain dynamic tokens.
    """
    parsed = urlparse(url)
    segments = [seg for seg in parsed.path.split("/") if seg]
    segments = segments[:-1]  # drop filename
    return "/" + "/".join(segments[-3:])


def extract_full_path(url: str) -> str:
    """Return full URL path or '/' if empty."""
    return urlparse(url).path or "/"


def get_registered_domain(url: str) -> str:
    """
    Extract registered domain (e.g., example.com).

    Used to ensure comparisons only occur within the same root domain.
    """
    return tldextract.extract(url).registered_domain


# ---------------------------------------------------------------------------
# FQDN EXTRACTION
# ---------------------------------------------------------------------------

def get_fqdn(url: str) -> str:
    """Return full FQDN from URL."""
    ext = tldextract.extract(url)
    return ".".join(part for part in [ext.subdomain, ext.domain, ext.suffix] if part)


def extract_fqdns(urls):
    """Extract unique FQDNs from a list of URLs."""
    return sorted(set(get_fqdn(u) for u in urls))


# ---------------------------------------------------------------------------
# DOMAIN STATISTICS (Top-K frequency features)
# ---------------------------------------------------------------------------

def get_top_tlds(fqdns, top_k):
    """Most common TLDs."""
    tlds = [extract(f).suffix for f in fqdns if f]
    return Counter(tlds).most_common(top_k)


def get_top_slds(fqdns, top_k):
    """Most common second-level domains."""
    slds = [extract(f).domain for f in fqdns if f]
    return Counter(slds).most_common(top_k)


def get_top_base_domains(fqdns, top_k):
    """Most common registered domains."""
    base = [f"{extract(f).domain}.{extract(f).suffix}" for f in fqdns if f]
    return Counter(base).most_common(top_k)


def get_top_num_labels(fqdns, top_k):
    """Distribution of number of labels in FQDN."""
    labels = [len(f.strip('.').split('.')) for f in fqdns if f]
    return Counter(labels).most_common(top_k)


def get_top_fqdn_lengths(fqdns, top_k):
    """Distribution of FQDN string lengths."""
    lengths = [len(f.strip()) for f in fqdns if f]
    return Counter(lengths).most_common(top_k)


# ---------------------------------------------------------------------------
# SUBDOMAIN / DIGIT FEATURES
# ---------------------------------------------------------------------------

def get_top_subdomain_lengths(fqdns, top_k):
    """Distribution of subdomain lengths."""
    lengths = [len(extract(f).subdomain) for f in fqdns if extract(f).subdomain]
    return Counter(lengths).most_common(top_k)


def get_top_digit_counts_fqdn(fqdns, top_k):
    """Digit frequency within full FQDN."""
    counts = [sum(c.isdigit() for c in f) for f in fqdns if f]
    return Counter(counts).most_common(top_k)


# ---------------------------------------------------------------------------
# ENTROPY & STRUCTURAL FEATURES
# ---------------------------------------------------------------------------

def shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of a string."""
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * log2(p) for p in probs)


def subdomain_shape(sub: str) -> str:
    """
    Classify subdomain structure into coarse shape categories.
    """
    if "-" in sub:
        return "dash-separated"
    if re.match(r"^[a-zA-Z]+\d+$", sub):
        return "alpha+digits"
    if len(sub) > 15:
        return "long_random"
    return "simple"


def get_top_subdomain_shapes(fqdns, top_k):
    """Distribution of subdomain structural shapes."""
    shapes = [subdomain_shape(extract(f).subdomain) for f in fqdns if extract(f).subdomain]
    return Counter(shapes).most_common(top_k)


def get_top_entropy_buckets(fqdns, top_k, bucket_size=0.5):
    """
    Bucket FQDN entropy into ranges (e.g., 3.0–3.5).
    """
    entropies = [shannon_entropy(f) for f in fqdns if f]
    buckets = [
        f"{int(e // bucket_size) * bucket_size:.1f}-{(int(e // bucket_size)+1)*bucket_size:.1f}"
        for e in entropies
    ]
    return Counter(buckets).most_common(top_k)


# ---------------------------------------------------------------------------
# STRUCTURAL SIGNATURES (used for rule generation)
# ---------------------------------------------------------------------------

def fqdn_structure_signature(fqdn: str) -> str:
    """
    Convert FQDN into structural token signature:
      digits -> #
      letters -> a
      mixed   -> a#
    """
    ext = extract(fqdn)
    tokens = re.split(r"[\.-]", ".".join(filter(None, [ext.subdomain, ext.domain, ext.suffix])))

    structure = []
    for token in tokens:
        if re.fullmatch(r"\d+", token):
            structure.append("#" * len(token))
        elif re.fullmatch(r"[a-zA-Z]+", token):
            structure.append("a" * len(token))
        elif re.fullmatch(r"[a-zA-Z]*\d+[a-zA-Z]*", token):
            structure.append("a#")
        else:
            structure.append("?")

    return "-".join(structure)


def get_top_structure_patterns(fqdns, top_k):
    """Most common structural signatures."""
    patterns = [fqdn_structure_signature(f) for f in fqdns]
    return Counter(patterns).most_common(top_k)


def pattern_to_regex(pattern: str) -> str:
    """
    Convert structural signature into regex pattern.
    """
    token_map = {"#": r"\d", "a": r"[a-z]", "?": r"[a-z0-9]"}

    regex_parts = []
    for token in pattern.split("-"):
        if all(c == token[0] for c in token):
            regex_parts.append(f"{token_map.get(token[0], '.')}" + f"{{{len(token)}}}")
        else:
            regex_parts.append(r"[a-z0-9]+")

    return "^" + "-".join(regex_parts) + "$"
