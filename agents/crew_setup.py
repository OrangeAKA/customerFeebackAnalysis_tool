"""
CrewAI Agent Setup for Feedback Analysis

Defines the multi-agent crew for context-aware root cause analysis
of customer feedback.
"""

import time
from typing import Optional, Any, Dict, List
from dataclasses import dataclass


@dataclass
class AgentProgress:
    """Track progress of agent execution."""
    agent_name: str
    status: str  # "pending", "running", "completed", "error"
    message: str = ""
    result: str = ""


class FeedbackAnalysisCrew:
    """
    Multi-agent crew for analyzing customer feedback with context
    from product documentation and release notes.
    """
    
    def __init__(
        self,
        groq_api_key: str,
        model: str = "llama-3.3-70b-versatile",
        docs_collection: Optional[Any] = None,
        releases_collection: Optional[Any] = None,
        progress_callback: Optional[callable] = None
    ):
        """
        Initialize the feedback analysis crew.
        
        Args:
            groq_api_key: Groq API key for LLM access
            model: Groq model to use
            docs_collection: ChromaDB collection for product docs
            releases_collection: ChromaDB collection for release notes
            progress_callback: Optional callback for progress updates
        """
        self.groq_api_key = groq_api_key
        self.model = model
        self.docs_collection = docs_collection
        self.releases_collection = releases_collection
        self.progress_callback = progress_callback
        
        self._llm = None
        self._agents = {}
        self._tools = {}
    
    def _get_llm(self):
        """Lazy-load the LLM."""
        if self._llm is None:
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(
                model=self.model,
                api_key=self.groq_api_key,
                temperature=0.1
            )
        return self._llm
    
    def _setup_tools(self):
        """Setup retrieval tools for agents."""
        from .tools import create_retrieval_tool
        
        if self.docs_collection is not None:
            self._tools['docs'] = create_retrieval_tool(
                self.docs_collection, 
                "product documentation"
            )
        
        if self.releases_collection is not None:
            self._tools['releases'] = create_retrieval_tool(
                self.releases_collection,
                "release notes"
            )
    
    def _setup_agents(self):
        """Create the CrewAI agents."""
        from crewai import Agent
        
        llm = self._get_llm()
        self._setup_tools()
        
        # Agent 1: Feedback Analyst
        self._agents['analyst'] = Agent(
            role="Feedback Analyst",
            goal="Analyze customer feedback to identify key themes, patterns, and critical issues",
            backstory=(
                "You are an expert product analyst who specializes in understanding "
                "customer feedback. You excel at identifying patterns, categorizing issues, "
                "and prioritizing problems based on their impact on users."
            ),
            llm=llm,
            max_iter=3,
            verbose=True
        )
        
        # Agent 2: Documentation Searcher
        doc_tools = [self._tools['docs']] if 'docs' in self._tools else []
        self._agents['doc_searcher'] = Agent(
            role="Documentation Specialist",
            goal="Search product documentation to find relevant information about reported issues",
            backstory=(
                "You are a documentation expert who knows how to efficiently search "
                "through product documentation to find relevant information. You help "
                "determine if issues are documented, known limitations, or unexpected bugs."
            ),
            llm=llm,
            tools=doc_tools,
            max_iter=3,
            verbose=True
        )
        
        # Agent 3: Release Correlator
        release_tools = [self._tools['releases']] if 'releases' in self._tools else []
        self._agents['correlator'] = Agent(
            role="Release Analyst",
            goal="Correlate customer complaints with recent releases and changelog entries",
            backstory=(
                "You are a release management expert who tracks product changes. "
                "You excel at connecting customer complaints to specific releases, "
                "identifying regression patterns, and spotting release-related issues."
            ),
            llm=llm,
            tools=release_tools,
            max_iter=3,
            verbose=True
        )
        
        # Agent 4: Synthesizer
        self._agents['synthesizer'] = Agent(
            role="Insights Synthesizer",
            goal="Combine all analysis into actionable recommendations for the product team",
            backstory=(
                "You are a senior product strategist who synthesizes complex information "
                "into clear, actionable insights. You create executive summaries that "
                "help product managers prioritize and address customer issues effectively."
            ),
            llm=llm,
            max_iter=3,
            verbose=True
        )
    
    def _update_progress(self, agent_name: str, status: str, message: str = ""):
        """Update progress via callback if available."""
        if self.progress_callback:
            self.progress_callback(AgentProgress(
                agent_name=agent_name,
                status=status,
                message=message
            ))
    
    def _rate_limit_delay(self):
        """Add delay to respect Groq rate limits."""
        time.sleep(2)  # 2 second delay between agent executions
    
    def analyze(
        self,
        feedback_summary: str,
        top_issues: List[str],
        category_breakdown: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Run the full analysis crew.
        
        Args:
            feedback_summary: Summary of the feedback data
            top_issues: List of top issues identified
            category_breakdown: Category counts from tagging
            
        Returns:
            Dictionary with analysis results from each agent
        """
        from crewai import Task, Crew
        
        self._setup_agents()
        
        results = {
            'feedback_analysis': '',
            'documentation_findings': '',
            'release_correlation': '',
            'final_report': '',
            'success': True,
            'error': None
        }
        
        try:
            # Prepare context
            issues_text = "\n".join(f"- {issue}" for issue in top_issues[:10])
            categories_text = "\n".join(
                f"- {cat}: {count}" for cat, count in category_breakdown.items()
            )
            
            context = f"""
## Feedback Summary
{feedback_summary}

## Top Issues Identified
{issues_text}

## Category Breakdown
{categories_text}
"""
            
            # Task 1: Analyze feedback
            self._update_progress("Feedback Analyst", "running", "Analyzing feedback patterns...")
            
            task1 = Task(
                description=f"""
Analyze the following customer feedback data and provide:
1. Key themes and patterns you observe
2. Most critical issues requiring immediate attention
3. Sentiment trends across categories
4. Any anomalies or unexpected patterns

{context}

Provide a structured analysis report.
""",
                expected_output="A detailed analysis of feedback themes, critical issues, and patterns",
                agent=self._agents['analyst']
            )
            
            # Task 2: Search documentation
            self._update_progress("Documentation Specialist", "pending")
            
            if 'docs' in self._tools:
                task2_desc = f"""
Based on the top issues from customer feedback, search the product documentation to:
1. Check if these issues are documented as known limitations
2. Find relevant documentation about the affected features
3. Identify any documentation gaps

Top issues to investigate:
{issues_text}

Use the search_documents tool to find relevant documentation.
"""
            else:
                task2_desc = """
No product documentation was provided for this analysis.
Please note that documentation search was skipped due to missing documentation uploads.
Provide a brief note about the importance of documentation review.
"""
            
            task2 = Task(
                description=task2_desc,
                expected_output="Documentation findings and gaps related to reported issues",
                agent=self._agents['doc_searcher']
            )
            
            # Task 3: Correlate with releases
            self._update_progress("Release Analyst", "pending")
            
            if 'releases' in self._tools:
                task3_desc = f"""
Investigate whether customer complaints correlate with recent releases:
1. Search release notes for changes related to reported issues
2. Identify potential regressions introduced by recent releases
3. Find patterns between complaint timing and release dates

Top issues to correlate:
{issues_text}

Use the search_documents tool to find relevant release notes.
"""
            else:
                task3_desc = """
No release notes were provided for this analysis.
Please note that release correlation was skipped due to missing release notes uploads.
Provide a brief note about the importance of release tracking.
"""
            
            task3 = Task(
                description=task3_desc,
                expected_output="Correlation analysis between issues and recent releases",
                agent=self._agents['correlator']
            )
            
            # Task 4: Synthesize findings
            self._update_progress("Insights Synthesizer", "pending")
            
            task4 = Task(
                description="""
Synthesize all findings from the team into an executive report:
1. Executive summary (2-3 sentences)
2. Key findings from feedback analysis
3. Documentation status and gaps
4. Release correlation insights
5. Prioritized recommendations (top 5)
6. Suggested next steps for the product team

Create a clear, actionable report suitable for product managers.
""",
                expected_output="Executive report with prioritized recommendations",
                agent=self._agents['synthesizer'],
                context=[task1, task2, task3]
            )
            
            # Create and run crew
            crew = Crew(
                agents=[
                    self._agents['analyst'],
                    self._agents['doc_searcher'],
                    self._agents['correlator'],
                    self._agents['synthesizer']
                ],
                tasks=[task1, task2, task3, task4],
                verbose=True
            )
            
            # Execute with progress updates
            self._update_progress("Feedback Analyst", "running", "Analyzing patterns...")
            self._rate_limit_delay()
            
            crew_result = crew.kickoff()
            
            # Extract results
            results['final_report'] = str(crew_result)
            
            # Mark all as completed
            for agent_name in ["Feedback Analyst", "Documentation Specialist", 
                             "Release Analyst", "Insights Synthesizer"]:
                self._update_progress(agent_name, "completed")
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            self._update_progress("Error", "error", str(e))
        
        return results
    
    def quick_analyze(self, feedback_text: str) -> str:
        """
        Perform a quick single-agent analysis without full crew.
        Useful for testing or when rate limits are a concern.
        
        Args:
            feedback_text: Raw feedback text to analyze
            
        Returns:
            Analysis result string
        """
        try:
            llm = self._get_llm()
            
            prompt = f"""Analyze this customer feedback and provide:
1. Key themes (bullet points)
2. Critical issues requiring attention
3. Top 3 recommendations

Feedback:
{feedback_text[:3000]}  # Limit to avoid token limits

Be concise and actionable."""
            
            response = llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"
