"""
Resume ingestion: load from bronze layer (data/bronze/resume.json).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_RESUME_PATH = _PROJECT_ROOT / "data" / "bronze" / "resume.json"


class ResumeLoader:
    """Load and process resume data from bronze layer."""

    def __init__(self, resume_path: Optional[str] = None):
        if resume_path is None:
            resume_path = _DEFAULT_RESUME_PATH
        self.resume_path = Path(resume_path)
        self.resume_data = self._load_resume()

    def _load_resume(self) -> Dict:
        if not self.resume_path.exists():
            raise FileNotFoundError(f"Resume file not found: {self.resume_path}")
        with open(self.resume_path, "r") as f:
            return json.load(f)

    def get_profile_summary(self) -> str:
        name = self.resume_data.get("name", "Unknown")
        title = self.resume_data.get("title", "")
        return f"{name} - {title}"

    def get_skills_text(self) -> str:
        skills = self.resume_data.get("skills", {})
        sorted_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
        top_skills = [skill for skill, _ in sorted_skills[:15]]
        return ", ".join(top_skills)

    def get_skills_dict(self) -> Dict[str, int]:
        return self.resume_data.get("skills", {})

    def get_projects(self) -> List[Dict]:
        return self.resume_data.get("projects", [])

    def get_experience(self) -> List[Dict]:
        return self.resume_data.get("experience", [])

    def get_target_roles(self) -> List[str]:
        return self.resume_data.get("target_roles", [])

    def get_target_salary(self) -> Dict:
        return self.resume_data.get("target_rate_range", {})

    def get_certifications(self) -> List[str]:
        return self.resume_data.get("certifications", [])

    def get_resume_text(self) -> str:
        parts = [
            self.get_profile_summary(),
            f"Skills: {self.get_skills_text()}",
        ]
        for exp in self.get_experience():
            company = exp.get("company", "")
            title = exp.get("title", "")
            keywords = ", ".join(exp.get("keywords", []))
            parts.append(f"{title} at {company}. Technologies: {keywords}")
        for proj in self.get_projects():
            name = proj.get("name", "")
            desc = proj.get("description", "")
            tech = ", ".join(proj.get("tech", []))
            parts.append(f"Project: {name}. {desc}. Tech: {tech}")
        certs = self.get_certifications()
        if certs:
            parts.append(f"Certifications: {', '.join(certs)}")
        return " ".join(parts)

    def get_contact_info(self) -> Dict:
        return self.resume_data.get("contact", {})

    def format_salary_preference(self) -> str:
        salary = self.get_target_salary()
        if not salary:
            return "Not specified"
        min_rate = salary.get("min", 0)
        max_rate = salary.get("max", 0)
        currency = salary.get("currency", "USD")
        unit = salary.get("unit", "hour")
        if min_rate == 0 and max_rate == 0:
            return "Not specified"
        return f"${min_rate}-${max_rate} {currency}/{unit}"
