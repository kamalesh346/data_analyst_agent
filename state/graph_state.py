from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    """
    Shared state contract for the LangGraph pipeline.

    Member 1 (Profiler Agent):
        - Reads:  csv_path
        - Writes: profile, profile_report_path, error_log, status

    Member 2 (Analysis Agent):
        - Reads:  csv_path, profile, profile_report_path
        - Writes: analysis_plan, analysis_results, generated_files, execution_log, reflection_notes
    """

    # Member 1 / Shared Core
    csv_path: str                          # input: path to uploaded CSV
    profile: Optional[Dict[str, Any]]      # structured dataset profile dict
    profile_report_path: Optional[str]     # absolute path to ydata-profiling HTML report
    error_log: List[str]                   # list of error messages (appended, never cleared)
    status: str                            # "running" | "completed" | "failed"

    # Member 2 (Analysis Agent) fields
    analysis_plan: Optional[List[Dict[str, Any]]]
    analysis_results: Optional[Dict[str, Any]]
    generated_files: Optional[List[str]]
    execution_log: Optional[List[Dict[str, Any]]]
    reflection_notes: Optional[List[str]]

    # Member 3 (Insight & Report Agent) fields
    validation_report: Optional[Dict[str, Any]]
    insights: Optional[List[Dict[str, Any]]]
    recommendations: Optional[List[str]]
    report_path: Optional[str]
