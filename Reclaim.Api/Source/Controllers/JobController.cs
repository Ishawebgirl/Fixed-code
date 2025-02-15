using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Reclaim.Api.Model;
using Reclaim.Api.Services;

namespace Reclaim.Api.Controllers;

[RequireRole]
[ValidateModel]
public class JobController : BaseController
{
    private readonly JobService _jobService;
    private readonly SearchService _searchService;

    public JobController(JobService jobService, SearchService searchService)
    {
        _jobService = jobService;
        _searchService = searchService;
    }

    /// <summary>
    /// Retrieve all scheduled jobs
    /// </summary>
    [HttpGet("jobs")]
    [RequireRole(Role.Administrator)]
    public async Task<List<Dtos.Job>> GetAllJobs()
    {
        var jobs = await _jobService.GetAll();

        return jobs.Select(x => new Dtos.Job(x)).ToList();
    }

    /// <summary>
    /// Run all pending jobs
    /// </summary>
    /// <remarks>
    /// ErrorCode.ScheduledJobTimeout
    /// </remarks>
    [HttpPost("jobs/runpending")]
    [RequireRole(Role.Administrator)]
    public async Task RunPendingJobs()
    {
        await _jobService.RunPending();
    }
}