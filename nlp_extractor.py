"""
nlp_extractor.py
────────────────
spaCy-based NLP engine.
Scenario text se extract karta hai:
  - classes     (Nouns / proper nouns)
  - attributes  (Adjectives + noun modifiers)
  - methods     (Verbs in base/present form)
  - actors      (Subject nouns — who does the action)
  - actions     (Verbs paired with actors)
  - objects     (Object nouns — what is acted upon)
  - interactions (subject → verb → object triples for Sequence diagram)
  - relationships (ENHANCED: strict UML type classification)

CHANGELOG (dataset-aligned update):
  - extract() now includes "relationships" with strict type classification
  - Compound noun extraction added: adjacent NOUN+NOUN pairs
  - STOPWORD_NOUNS/VERBS expanded
"""

import re
import spacy
from typing import Dict, List, Any, Set

# ── Load spaCy model (LAZY) ────────────────────────────────────────────────────
# FIX: module-level spacy.load() crashed the whole app at import time if the
# model wasn't installed. Now loaded lazily on first use so a missing model
# only disables the NLP feature instead of taking down the entire backend.
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
    return _nlp

# ── Words to ignore ──────────────────────────────────────────────────────────
STOPWORD_NOUNS = {
    "system", "thing", "way", "time", "part", "kind",  # NOTE: "user" removed — it IS a valid participant
    "number", "lot", "place", "case", "week", "month", "year",
    "day", "point", "group", "problem", "fact", "example", "set",
    "information", "management", "area", "period", "status",
    "type", "level", "rate", "amount", "service", "process",
    "detail", "result", "item", "list", "record",
}

STOPWORD_VERBS = {
    "be", "is", "are", "was", "were", "have", "has", "had",
    "do", "does", "did", "say", "get", "make", "go", "know",
    "take", "see", "come", "think", "look", "want", "give",
    "use", "find", "tell", "ask", "seem", "feel", "try", "leave",
    "call", "keep", "let", "begin", "show", "hear", "play", "run",
    "move", "live", "believe", "hold", "bring", "happen", "write",
    "provide", "set", "put", "mean", "might", "need", "start", "turn",
    "belong", "contain", "offer", "enroll", "accommodate",
    "operate", "conduct", "host", "consist", "employ", "manage",
    "associate", "relate", "connect", "link", "include",
    "receive", "send", "submit", "forward",  # FIX: common sequence noise verbs
}

# ── Relationship type priorities ────────────────────────────────────────────
REL_TYPE_PRIORITY = {
    "composition": 4,
    "aggregation": 3,
    "generalization": 2,
    "association": 1,
}

# ── Compound class seeds ─────────────────────────────────────────────────────
COMPOUND_CLASS_SEEDS: List[tuple] = [
    (r"\bblood\s+group\b",          "BloodGroup"),
    (r"\bblood\s+unit\b",           "BloodUnit"),
    (r"\bblood\s+bank\b",           "BloodBank"),
    (r"\btour\s+package\b",         "TourPackage"),
    (r"\btravel\s+agenc\w+\b",      "TravelAgency"),
    (r"\bcourier\s+compan\w+\b",    "CourierCompany"),
    (r"\bservice\s+plan\b",         "ServicePlan"),
    (r"\btelecom\s+provider\b",     "TelecomProvider"),
    (r"\brelief\s+camp\b",          "ReliefCamp"),
    (r"\bdisaster\s+(?:management\s+)?system\b", "DisasterSystem"),
    (r"\bsports\s+academ\w+\b",     "SportsAcademy"),
    (r"\bmusic\s+academ\w+\b",      "MusicAcademy"),
    (r"\bflight\s+training\s+academ\w+\b", "Academy"),
    (r"\bonline\s+forum\b",         "OnlineForum"),
    (r"\bdiscussion\s+thread\b",    "Thread"),
    (r"\bservice\s+request\b",      "ServiceRequest"),
    (r"\bleave\s+request\b",        "LeaveRequest"),
    (r"\bsupport\s+agent\b",        "SupportAgent"),
    (r"\btransport\s+compan\w+\b",  "TransportCompany"),
    (r"\bparking\s+area\b",         "ParkingArea"),
    (r"\bshopping\s+mall\b",        "ShoppingMall"),
    (r"\bshopping\s+platform\b",    "Platform"),
    (r"\bconstruction\s+compan\w+\b", "ConstructionCompany"),
    (r"\bresearch\s+institute\b",   "ResearchInstitute"),
    (r"\binspection\s+center\b",    "InspectionCenter"),
    (r"\binsurance\s+compan\w+\b",  "InsuranceCompany"),
    (r"\blogistics\s+compan\w+\b",  "LogisticsCompany"),
    (r"\bdelivery\s+van\b",         "DeliveryVan"),
    (r"\bhealth\s+policy\b",        "HealthPolicy"),
    (r"\bvehicle\s+policy\b",       "VehiclePolicy"),
    (r"\bsavings\s+account\b",      "SavingsAccount"),
    (r"\bcurrent\s+account\b",      "CurrentAccount"),
    (r"\bpermanent\s+faculty\b",    "PermanentFaculty"),
    (r"\bvisiting\s+faculty\b",     "VisitingFaculty"),
    (r"\bfaculty\s+member\b",       "FacultyMember"),
    (r"\bundergraduate\s+student\b","UndergraduateStudent"),
    (r"\bpostgraduate\s+student\b", "PostgraduateStudent"),
    (r"\btechnical\s+staff\b",      "TechnicalStaff"),
    (r"\badministrative\s+staff\b", "AdministrativeStaff"),
    (r"\btransport\s+service\b",    "TransportService"),
    (r"\butility\s+service\b",      "UtilityService"),
    (r"\bdigital\s+media\b",        "DigitalMedia"),
    (r"\bcity\s+employee\b",        "CityEmployee"),
]


def _normalize(text: str) -> str:
    """Lowercase + strip."""
    return text.strip().lower()


def _to_pascal_case(word: str) -> str:
    """login_user → LoginUser,  admin → Admin"""
    return "".join(part.capitalize() for part in word.replace("_", " ").split())


class NLPExtractor:
    """
    One class handles all three diagram types.
    Call extract(scenario) to get all elements at once.
    """

    def extract(self, scenario: str) -> Dict[str, Any]:
        doc = _get_nlp()(scenario)

        classes    = self._extract_classes(doc)
        attributes = self._extract_attributes(doc)
        methods    = self._extract_methods(doc)
        actors     = self._extract_actors(doc)
        actions    = self._extract_actions(doc)
        objects    = self._extract_objects(doc)
        interactions = self._extract_interactions(doc)

        # ── Compound class extraction via COMPOUND_CLASS_SEEDS ────────────────
        compound_classes = self._extract_compound_classes(scenario)
        all_classes = sorted(set(classes) | set(compound_classes))

        # ── **NEW: ENHANCED RELATIONSHIP EXTRACTION** ─────────────────────────
        relationships = self._extract_enhanced_relationships(doc, scenario, all_classes)

        return {
            # Raw scenario text — required by ClassDiagramValidator
            "scenario":     scenario,
            "raw_text":     scenario,   # alias for backward-compat
            # Class Diagram
            "classes":      all_classes,
            "attributes":   sorted(set(attributes)),
            "methods":      sorted(set(methods)),
            # Use Case Diagram
            "actors":       sorted(set(actors)),
            "actions":      sorted(set(actions)),
            # Sequence Diagram
            "objects":      sorted(set(objects)),
            "interactions": interactions,
            # **NEW: STRICT UML RELATIONSHIPS**
            "relationships": relationships,
        }

    def _extract_enhanced_relationships(self, doc, scenario: str, classes: List[str]) -> List[Dict[str, Any]]:
        """
        Extract relationships with STRICT UML type classification.
        Format: [{"from": "ClassA", "to": "ClassB", "type": "generalization|association|aggregation|composition"}]
        """
        class_set = {c.lower() for c in classes}
        relationships: List[Dict[str, Any]] = []
        
        # Sentence-level regex patterns (strict matching)
        sentences = re.split(r"[.!?;]+", scenario)
        patterns = [
            # COMPOSITION (highest priority)
            (r"\b([A-Z][a-zA-Z]*)\s+(?:consists|composed|strictly owns|owns)\s+(?:of\s+)?([A-Z][a-zA-Z]*)", "composition"),
            (r"\b([A-Z][a-zA-Z]*)\s+cannot\s+exist\s+without\s+([A-Z][a-zA-Z]*)", "composition"),
            
            # AGGREGATION
            (r"\b([A-Z][a-zA-Z]*)\s+(?:contains|holds|collection|made up of)\s+([A-Z][a-zA-Z]*)", "aggregation"),
            
            # GENERALIZATION
            (r"\b([A-Z][a-zA-Z]*)\s+(?:is\s+a\s+(?:type|kind)\s+of|inherits|extends)\s+([A-Z][a-zA-Z]*)", "generalization"),
            (r"\b([A-Z][a-zA-Z]*)\s+is\s+an?\s+([A-Z][a-zA-Z]+)", "generalization"),
            
            # ASSOCIATION (lowest priority)
            (r"\b([A-Z][a-zA-Z]*)\s+(?:has|uses|manages|related|associated)\s+(?:with\s+)?([A-Z][a-zA-Z]*)", "association"),
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) < 3:
                continue
            
            for pattern_str, rel_type in patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                match = pattern.search(sentence)
                if match:
                    from_cls, to_cls = match.groups()
                    from_lower, to_lower = from_cls.lower(), to_cls.lower()
                    
                    # STRICT: Both must be known classes
                    if from_lower in class_set and to_lower in class_set and from_lower != to_lower:
                        pair_key = (from_lower, to_lower)
                        
                        # Keep highest priority match
                        existing_idx = next((i for i, r in enumerate(relationships) 
                                           if (r["from"].lower(), r["to"].lower()) == pair_key), -1)
                        
                        rel_priority = REL_TYPE_PRIORITY.get(rel_type, 0)
                        if existing_idx == -1 or rel_priority > REL_TYPE_PRIORITY.get(relationships[existing_idx]["type"], 0):
                            rel_dict = {
                                "from": from_cls,
                                "to": to_cls,
                                "type": rel_type,
                                "trigger": match.group(0)
                            }
                            if existing_idx != -1:
                                relationships[existing_idx] = rel_dict
                            else:
                                relationships.append(rel_dict)
        
        return relationships

    # ── Compound class extractor ──────────────────────────────────────────────
    def _extract_compound_classes(self, scenario: str) -> List[str]:
        found: List[str] = []
        for (pattern, class_name) in COMPOUND_CLASS_SEEDS:
            if re.search(pattern, scenario, re.IGNORECASE):
                found.append(class_name)
        return found

    # ─────────────────────────────────────────────────────────────────────────
    #  CLASS DIAGRAM elements (UNCHANGED)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_classes(self, doc) -> List[str]:
        classes = []
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                word = _normalize(token.lemma_)
                if word not in STOPWORD_NOUNS and len(word) > 2:
                    classes.append(_to_pascal_case(word))
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "PRODUCT", "WORK_OF_ART"):
                classes.append(_to_pascal_case(ent.text))
        return list(set(classes))

    def _extract_attributes(self, doc) -> List[str]:
        attributes = []
        for token in doc:
            if token.pos_ == "ADJ" and token.head.pos_ in ("NOUN", "PROPN"):
                attr = _normalize(token.text)
                if len(attr) > 2:
                    attributes.append(attr)
            if token.dep_ == "compound" and token.head.pos_ in ("NOUN", "PROPN"):
                attr = _normalize(token.text + "_" + token.head.text)
                attributes.append(attr)
        return list(set(attributes))

    def _extract_methods(self, doc) -> List[str]:
        methods = []
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                lemma = _normalize(token.lemma_)
                if lemma not in STOPWORD_VERBS and len(lemma) > 2:
                    methods.append(lemma)
        return list(set(methods))

    # ─────────────────────────────────────────────────────────────────────────
    #  USE CASE DIAGRAM elements (UNCHANGED)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_actors(self, doc) -> List[str]:
        actors = []
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass") and token.pos_ in ("NOUN", "PROPN"):
                word = _normalize(token.lemma_)
                if word not in STOPWORD_NOUNS and len(word) > 2:
                    actors.append(_to_pascal_case(word))
        return list(set(actors))

    def _extract_actions(self, doc) -> List[str]:
        actions = []
        for token in doc:
            if token.pos_ == "VERB":
                has_subject = any(
                    child.dep_ in ("nsubj", "nsubjpass")
                    for child in token.children
                )
                if has_subject:
                    lemma = _normalize(token.lemma_)
                    if lemma not in STOPWORD_VERBS and len(lemma) > 2:
                        actions.append(lemma)
        return list(set(actions))

    # ─────────────────────────────────────────────────────────────────────────
    #  SEQUENCE DIAGRAM elements (UNCHANGED)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_objects(self, doc) -> List[str]:
        objects = []
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                word = _normalize(token.lemma_)
                if word not in STOPWORD_NOUNS and len(word) > 2:
                    objects.append(_to_pascal_case(word))
        return list(set(objects))

    def _extract_interactions(self, doc) -> List[Dict[str, str]]:
        interactions = []
        for token in doc:
            if token.pos_ != "VERB":
                continue
            lemma = _normalize(token.lemma_)
            if lemma in STOPWORD_VERBS:
                continue

            subjects = [
                c for c in token.children
                if c.dep_ in ("nsubj", "nsubjpass") and c.pos_ in ("NOUN", "PROPN")
            ]
            dobjects = [
                c for c in token.children
                if c.dep_ in ("dobj", "pobj", "attr") and c.pos_ in ("NOUN", "PROPN")
            ]

            for subj in subjects:
                for obj in dobjects:
                    s = _normalize(subj.lemma_)
                    o = _normalize(obj.lemma_)
                    if s not in STOPWORD_NOUNS and o not in STOPWORD_NOUNS:
                        interactions.append({
                            "from":    _to_pascal_case(s),
                            "message": lemma,
                            "to":      _to_pascal_case(o),
                        })
        return interactions
