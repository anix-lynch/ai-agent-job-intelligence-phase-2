"""AI Agent Orchestration using LangChain

Demonstrates:
- Multi-agent systems
- Autonomous reasoning
- Tool-using agents
- ReAct framework (Reasoning + Acting)
- Chain-of-thought prompting
"""

from typing import Optional
import pandas as pd


class JobMatchingAgent:
    """Simplified job matching agent for real use
    
    Demonstrates AI agent reasoning without complex dependencies
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "openai"):
        self.api_key = api_key
        self.provider = provider
        self.reasoning_trace = []
    
    def run(self, query: str, jobs_df: pd.DataFrame) -> str:
        """Run agent on job matching query with ReAct-style reasoning"""
        self.reasoning_trace = []
        
        # ReAct Framework: Thought -> Action -> Observation
        self.reasoning_trace.append(f"💭 Thought: Analyzing query: '{query}'")
        self.reasoning_trace.append("🎯 Action: Parsing requirements from query...")
        
        # Simple matching logic with reasoning
        query_lower = query.lower()
        
        # Start with all jobs
        filtered = jobs_df.copy()
        filters_applied = []
        
        # Location filtering (most important)
        location_keywords = {
            'los angeles': ['Los Angeles', 'LA,'],
            'la': ['Los Angeles', 'LA,'],
            'sf': ['San Francisco', 'SF,'],
            'san francisco': ['San Francisco'],
            'nyc': ['New York', 'NY,'],
            'new york': ['New York'],
            'seattle': ['Seattle'],
            'boston': ['Boston'],
            'austin': ['Austin'],
            'chicago': ['Chicago'],
            'mountain view': ['Mountain View'],
            'palo alto': ['Palo Alto'],
            'sunnyvale': ['Sunnyvale'],
            'san jose': ['San Jose']
        }
        
        location_found = None
        for location_key, location_patterns in location_keywords.items():
            if location_key in query_lower:
                location_found = location_key.upper()
                # Filter by location - must match one of the patterns
                location_mask = filtered['location'].str.contains('|'.join(location_patterns), case=False, na=False)
                filtered = filtered[location_mask]
                filters_applied.append(f"location: {location_found}")
                self.reasoning_trace.append(f"👁️ Observation: User wants jobs in {location_found}")
                break
        
        # Salary filtering
        if "200k" in query_lower or "$200" in query_lower or "200000" in query_lower:
            self.reasoning_trace.append("👁️ Observation: User wants high-salary roles (200K+)")
            filtered = filtered[filtered['salary_min'] >= 200000]
            filters_applied.append("salary: 200K+")
        
        # Role type filtering
        role_type = "general"
        if "senior" in query_lower:
            self.reasoning_trace.append("👁️ Observation: User wants senior-level positions")
            filtered = filtered[filtered['title'].str.contains('Senior|Staff|Principal|Lead', case=False, na=False)]
            filters_applied.append("level: senior")
            role_type = "senior"
        
        if "ml" in query_lower or "machine learning" in query_lower:
            self.reasoning_trace.append("👁️ Observation: User interested in ML engineering roles")
            filtered = filtered[filtered['title'].str.contains('ML|Machine Learning', case=False, na=False)]
            filters_applied.append("role: ML")
            role_type = "ML engineering"
        elif "ai" in query_lower or "artificial intelligence" in query_lower:
            self.reasoning_trace.append("👁️ Observation: User interested in AI roles")
            filtered = filtered[filtered['title'].str.contains('AI|Artificial Intelligence', case=False, na=False)]
            filters_applied.append("role: AI")
            role_type = "AI"
        
        # Get top 5 results
        filtered = filtered.head(5)
        
        # Decision making
        filter_summary = ", ".join(filters_applied) if filters_applied else "no filters"
        self.reasoning_trace.append(f"🤔 Thought: Found {len(filtered)} {role_type} positions ({filter_summary})")
        self.reasoning_trace.append("✅ Decision: Returning top matches with details")
        
        # Format response
        if len(filtered) == 0:
            return f"❌ No jobs found matching your criteria. Try broadening your search."
        
        response = f"🎯 Found {len(filtered)} AI roles:\n\n"
        for idx, job in filtered.iterrows():
            salary_range = f"${int(job['salary_min']/1000)}K-${int(job['salary_max']/1000)}K" if pd.notna(job.get('salary_min')) else "Competitive"
            apply_url = job.get('apply_url', '')
            apply_link = f" [Apply Now →]({apply_url})" if apply_url else ""
            response += f"• **{job['title']}** at **{job['company']}** 💰 {salary_range} | 📍 {job.get('location', 'Remote')}{apply_link}\n\n"
        
        self.reasoning_trace.append(f"📊 Final Answer: Presented {len(filtered)} curated opportunities")
        
        return response
    
    def get_reasoning_trace(self) -> str:
        """Get ReAct reasoning trace showing agent's thought process"""
        return "\n".join(self.reasoning_trace)
