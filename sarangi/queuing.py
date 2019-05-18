from subprocess import Popen, PIPE
import getpass

__all__ = ['load_jobs_PBS', 'get_queued_jobs_PBS', 'get_queued_jobs_SLURM', 'get_queued_jobs']


def load_jobs_PBS():
    import xml.etree.ElementTree as ET
    process = Popen(['qstat', '-x',], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    jobs = []
    root = ET.fromstring(stdout)
    for node in root:
       job = {}
       for pair in node:
           if pair.tag in ['Job_Name', 'Job_Owner', 'job_state']:
               job[pair.tag] = pair.text
       jobs.append(job)
    return jobs


def get_queued_jobs_PBS():
    user = getpass.getuser()
    names = []
    for job in load_jobs_PBS():
        if job['Job_Owner'][0:len(user)] == user:
            names.append(job['Job_Name'])
    return names


def get_queued_jobs_SLURM():
    user = getpass.getuser()
    process = Popen(['squeue', '-o', '%j', '-h', '-u', user], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    return [j.strip() for j in stdout]


def get_queued_jobs():
    try:
        return get_queued_jobs_SLURM()
    except FileNotFoundError:
        try:
            return get_queued_jobs_PBS()
        except FileNotFoundError:
            return None