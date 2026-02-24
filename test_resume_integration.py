"""
Test Resume MCP Integration
Verify that resume data loads correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.ingestion import ResumeLoader

def test_resume_loading():
    """Test resume loading functionality"""
    print("=" * 80)
    print("Testing Resume MCP Integration")
    print("=" * 80)
    
    try:
        # Load resume
        print("\n1. Loading resume from data/bronze/resume.json...")
        resume = ResumeLoader()
        print("✅ Resume loaded successfully!")
        
        # Test profile summary
        print("\n2. Profile Summary:")
        print(f"   {resume.get_profile_summary()}")
        
        # Test skills
        print("\n3. Top Skills:")
        print(f"   {resume.get_skills_text()}")
        
        # Test target roles
        print("\n4. Target Roles:")
        for role in resume.get_target_roles():
            print(f"   - {role}")
        
        # Test salary preference
        print("\n5. Salary Preference:")
        print(f"   {resume.format_salary_preference()}")
        
        # Test experience
        print("\n6. Experience:")
        for exp in resume.get_experience():
            print(f"   - {exp['title']} at {exp['company']} ({exp['duration']})")
        
        # Test projects
        print("\n7. Projects:")
        for proj in resume.get_projects():
            print(f"   - {proj['name']} (Weight: {proj['weight']}/10)")
        
        # Test certifications
        print("\n8. Certifications:")
        for cert in resume.get_certifications():
            print(f"   - {cert}")
        
        # Test full resume text
        print("\n9. Full Resume Text (first 500 chars):")
        resume_text = resume.get_resume_text()
        print(f"   {resume_text[:500]}...")
        print(f"   Total length: {len(resume_text)} characters")
        
        # Test contact info
        print("\n10. Contact Information:")
        contact = resume.get_contact_info()
        for key, value in contact.items():
            print(f"   - {key}: {value}")
        
        print("\n" + "=" * 80)
        print("✅ All tests passed! Resume MCP integration is working correctly.")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_resume_loading()
    sys.exit(0 if success else 1)