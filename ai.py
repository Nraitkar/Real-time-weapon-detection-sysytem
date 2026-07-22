# Job Scheduling Problem using Greedy Algorithm

def job_scheduling(jobs):
    # Sort jobs according to profit (descending order)
    jobs.sort(key=lambda x: x[2], reverse=True)

    # Find maximum deadline
    max_deadline = max(job[1] for job in jobs)

    # Initialize slots
    slots = [False] * max_deadline
    result = ['-1'] * max_deadline

    total_profit = 0

    # Schedule jobs
    for job in jobs:
        job_id, deadline, profit = job

        # Find a free slot from deadline-1 to 0
        for j in range(min(max_deadline, deadline) - 1, -1, -1):
            if not slots[j]:
                slots[j] = True
                result[j] = job_id
                total_profit += profit
                break

    return result, total_profit


# Driver Code
jobs = [
    ('J1', 2, 100),
    ('J2', 1, 19),
    ('J3', 2, 27),
    ('J4', 1, 25),
    ('J5', 3, 15)
]

scheduled_jobs, profit = job_scheduling(jobs)

print("Scheduled Jobs:")
for job in scheduled_jobs:
    print(job)

print("Total Profit:", profit)

       
           