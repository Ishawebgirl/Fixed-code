using Microsoft.AspNetCore.SignalR;
using Microsoft.EntityFrameworkCore;
using Reclaim.Api.Model;

namespace Reclaim.Api.Services;

public class JobService
{
    private readonly DatabaseContext _db;
    private readonly EmailService _emailService;
    private readonly LogService _logService;
    private readonly IHubContext<Hubs.JobHub> _hubContext;

    public JobService(DatabaseContext db, EmailService emailService, LogService logService, IHubContext<Hubs.JobHub> hubContext)
    {
        _db = db;
        _emailService = emailService;
        _logService = logService;
        _hubContext = hubContext;
    }

    public async Task<List<Job>> GetAll()
    {
        return await _db.Jobs
            .ToListAsync();
    }

    public async Task RunPending()
    {
        // lock (this)
        {
            var pendingJobs = await _db.Jobs.Where(
                x => x.Status == JobStatus.Ready
                && DateTime.UtcNow >= x.NextEvent).ToListAsync();

            foreach (var job in pendingJobs)
                await Run(job);
        }
    }

    public async Task Run(Job job)
    {
        var now = DateTime.UtcNow;
        var jobEvent = new JobEvent { JobID = job.ID, StartedTimestamp = now };

        job.Status = JobStatus.Running;
        await _db.SaveChangesAsync();
        await SendStatusUpdate(job);

        var task = Task.Delay(job.Timeout * 1000)
            .ContinueWith(async x =>
            {
                using (var db = new DatabaseContext(DbContextOptions.Get()))  // request-scoped dbcontext is already gone by the time this task executes
                {
                    var reloaded = await db.Jobs.FirstAsync(x => x.ID == job.ID);

                    if (reloaded.Status != JobStatus.Ready)
                    {
                        db.JobEvents.Add(new JobEvent { JobID = job.ID, ItemCount = 0, StartedTimestamp = now, TimedOutTimestamp = DateTime.UtcNow });
                        reloaded.Status = JobStatus.TimedOut;

                        await db.SaveChangesAsync();
                        await SendStatusUpdate(reloaded);

                        throw new ApiException(ErrorCode.ScheduledJobTimeout, $"Scheduled job {job.Name} did not complete within the expected {job.Interval}s interval.");
                    }
                }
            });

        try
        {
            switch (job.Type)
            {
                case JobType.RunDiagnostics:
                    await _emailService.Queue(EmailTemplateCode.Diagnostics, Setting.SystemAdministratorAccountID, DateTime.UtcNow);
                    jobEvent.ItemCount = 1;
                    break;

                case JobType.DeliverEmail:
                    jobEvent.ItemCount = await _emailService.DeliverPending();
                    break;
            }
        }
        catch (ApiException ex)
        {
            var logEntry = await _logService.Add(ex);
            jobEvent.LogEntryID = logEntry.ID;

            await _emailService.SendUnhandledException(ex);
        }
        catch (Exception ex)
        {
            var logEntry = await _logService.Add(ex);
            jobEvent.LogEntryID = logEntry.ID;

            await _emailService.SendUnhandledException(new ApiException(ex, ErrorCode.Unhandled));
        }

        var nextEvent = job.NextEvent;

        jobEvent.FinishedTimestamp = DateTime.UtcNow;

        while (nextEvent.CompareTo(jobEvent.FinishedTimestamp) <= 0)
            nextEvent = nextEvent.AddSeconds((double)job.Interval);

        job.Status = JobStatus.Ready;
        job.NextEvent = nextEvent;

        _db.JobEvents.Add(jobEvent);
        await _db.SaveChangesAsync();
        await SendStatusUpdate(job);
    }

    private async Task SendStatusUpdate(Job job)
    {
        await _hubContext.Clients.All.SendAsync("SetJobStatus", job.ID, job.Status.ToString(), job.NextEvent);
    }
}