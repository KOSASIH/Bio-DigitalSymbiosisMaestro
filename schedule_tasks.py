function schedule_tasks(tasks, ai_agents, human_agents):
    assignment_list = []

    # Sort tasks based on complexity
    sorted_tasks = sort_tasks_by_complexity(tasks)

    for task in sorted_tasks:
        best_agent = None
        best_score = -1

        for agent in ai_agents + human_agents:
            if agent.workload < agent.max_workload:
                score = calculate_suitability_score(agent, task)
                if score > best_score:
                    best_score = score
                    best_agent = agent

        if best_agent is not None:
            best_agent.workload += 1
            assignment_list.append((task, best_agent))

    return assignment_list

def calculate_suitability_score(agent, task):
    # Calculate suitability score based on expertise, workload, and other factors
    score = agent.expertise * (1 - agent.workload_factor) + task.complexity * task.priority_factor
    return score

def sort_tasks_by_complexity(tasks):
    # Sort tasks based on complexity in ascending order
    sorted_tasks = sorted(tasks, key=lambda x: x.complexity)
    return sorted_tasks
