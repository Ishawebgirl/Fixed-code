using Microsoft.EntityFrameworkCore;
using Reclaim.Api.Model;

namespace Reclaim.Api.Services;

public class StatusService
{
    private readonly DatabaseContext _db;
    private readonly CacheService _cacheService;

    public StatusService(DatabaseContext db, CacheService cacheService)
    {
        _db = db;
        _cacheService = cacheService;
    }

    public List<string> GetIssues()
    {
        var issues = new List<string>();
        var pausedJobs = _db.Jobs.Where(x => x.Status != JobStatus.Ready && x.Type != JobType.DeliverEmail);

        foreach (var job in pausedJobs)
            issues.Add($"The {job.Name} job is {job.Status.DisplayName()}");

        return issues;
    }
}
