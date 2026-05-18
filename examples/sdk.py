from chronicle import Chronicle

chronicle = Chronicle(repo_path="/Users/animeshdutta/Projects/Nudge_git/Nudge")
packet = chronicle.prepare_prompt_packet(
    query="How to improve latency of governance agent",
    token_budget=3000,
)
print(packet)