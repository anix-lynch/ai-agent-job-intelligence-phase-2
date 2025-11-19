"""
Resume MCP Integration
Loads personalized resume data for intelligent job matching
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

class ResumeLoader:
    """Load and process resume data from Resume MCP"""
    
    def __init__(self, resume_path: Optional[str] = None):
        """Initialize with resume data path"""
        if resume_path is None:
            # Default to data/resume.json
            resume_path = Path(__file__).parent.parent / "data" / "resume.json"
        
        self.resume_path = Path(resume_path)
        self.resume_data = self._load_resume()
    
    def _load_resume(self) -> Dict:
        """Load resume JSON data"""
        if not self.resume_path.exists():
            raise FileNotFoundError(f"Resume file not found: {self.resume_path}")
        
        with open(self.resume_path, 'r') as f:
            return json.load(f)
    
    def get_profile_summary(self) -> str:
        """Get formatted profile summary"""
        name = self.resume_data.get('name', 'Unknown')
        title = self.resume_data.get('title', '')
        
        return f"{name} - {title}"
    
    def get_skills_text(self) -> str:
        """Get skills as text for semantic search"""
        skills = self.resume_data.get('skills', {})
        
        # Sort by proficiency (value) and get top skills
        sorted_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)
        top_skills = [skill for skill, _ in sorted_skills[:15]]
        
        return ", ".join(top_skills)
    
    def get_skills_dict(self) -> Dict[str, int]:
        """Get skills dictionary with proficiency levels"""
        return self.resume_data.get('skills', {})
    
    def get_projects(self) -> List[Dict]:
        """Get projects list"""
        return self.resume_data.get('projects', [])
    
    def get_experience(self) -> List[Dict]:
        """Get experience list"""
        return self.resume_data.get('experience', [])
    
    def get_target_roles(self) -> List[str]:
        """Get target job roles"""
        return self.resume_data.get('target_roles', [])
    
    def get_target_salary(self) -> Dict:
        """Get target salary range"""
        return self.resume_data.get('target_rate_range', {})
    
    def get_certifications(self) -> List[str]:
        """Get certifications list"""
        return self.resume_data.get('certifications', [])
    
    def get_resume_text(self) -> str:
        """Get full resume as text for semantic matching"""
        parts = []
        
        # Add profile
        parts.append(self.get_profile_summary())
        
        # Add skills
        parts.append(f"Skills: {self.get_skills_text()}")
        
        # Add experience
        for exp in self.get_experience():
            company = exp.get('company', '')
            title = exp.get('title', '')
            keywords = ', '.join(exp.get('keywords', []))
            parts.append(f"{title} at {company}. Technologies: {keywords}")
        
        # Add projects
        for proj in self.get_projects():
            name = proj.get('name', '')
            desc = proj.get('description', '')
            tech = ', '.join(proj.get('tech', []))
            parts.append(f"Project: {name}. {desc}. Tech: {tech}")
        
        # Add certifications
        certs = self.get_certifications()
        if certs:
            parts.append(f"Certifications: {', '.join(certs)}")
        
        return " ".join(parts)
    
    def get_contact_info(self) -> Dict:
        """Get contact information"""
        return self.resume_data.get('contact', {})
    
    def format_salary_preference(self) -> str:
        """Format salary preference as string"""
        salary = self.get_target_salary()
        min_rate = salary.get('min', 0)
        max_rate = salary.get('max', 0)
        currency = salary.get('currency', 'USD')
        unit = salary.get('unit', 'hour')
        
        return f"${min_rate}-${max_rate} {currency}/{unit}"