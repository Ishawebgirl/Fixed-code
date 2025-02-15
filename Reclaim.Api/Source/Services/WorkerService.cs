namespace Reclaim.Api.Services;

public class WorkerService : BackgroundService
{
    private readonly IServiceScope _scope;

    public WorkerService(IServiceProvider services)
    {
        _scope = services.CreateScope(); // CreateScope is in Microsoft.Extensions.DependencyInjection
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            await Task.Delay(Setting.ScheduledJobPollingFrequency * 1000, stoppingToken);

            using (var scope = _scope.ServiceProvider.CreateAsyncScope())
            {
                var jobService = scope.ServiceProvider.GetService<JobService>();
                await jobService.RunPending();
            }
        }        
    }
}