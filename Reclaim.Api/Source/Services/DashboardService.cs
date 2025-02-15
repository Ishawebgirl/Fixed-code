using Microsoft.EntityFrameworkCore;
using Reclaim.Api.Model;

namespace Reclaim.Api.Services;

public class DashboardService
{
    private readonly DatabaseContext _db;
    private readonly CacheService _cacheService;

    public DashboardService(DatabaseContext db, CacheService cacheService)
    {
        _db = db;
        _cacheService = cacheService;
    }

    public async Task<Dtos.AdministratorDashboard> Get()
    {
        var dashboard = await _cacheService.Get("Dashboard-Administrator", 300, () =>
        {
            var result = new Dtos.AdministratorDashboard();

            result.UniqueSignins = new Dtos.DashboardAggregate(817, 909, Dtos.DashboardAggregate.Type.Integer, Dtos.DashboardAggregate.Period.Day);
            result.ClaimsValueUnderInvestigation = new Dtos.DashboardAggregate(599002.11m, 524191.95m, Dtos.DashboardAggregate.Type.Money, Dtos.DashboardAggregate.Period.Month);
            result.NewOrders = new Dtos.DashboardAggregate(15, 7, Dtos.DashboardAggregate.Type.Integer, Dtos.DashboardAggregate.Period.Month);
            result.MonthlyRevenue = new Dtos.DashboardAggregate(7881.15m, 5808.08m, Dtos.DashboardAggregate.Type.Money, Dtos.DashboardAggregate.Period.Month);

            result.ClaimsByState = _db.Claims
                .Include(x => x.Policy.State)
                .Where(x => x.Status != ClaimStatus.Tombstoned)
                .GroupBy(x => x.Policy.State.Code)
                .ToDictionary(x => x.Key, x => x.Count());
     
            result.ClaimsByMonth = new Dictionary<DateTime, List<Dtos.AdministratorDashboard.ClaimStatusValue>>();

            var yearAgo = new DateOnly(DateTime.Now.Year, DateTime.Now.Month, 1).AddMonths(-12);
            var claimsByMonth = _db.Claims
                .Where(x => x.Status != ClaimStatus.Tombstoned)
                .Where(x => x.EventDate >= yearAgo)
                .GroupBy(x => new { x.Status, x.EventDate.Month, x.EventDate.Year })
                .Select(x => new { x.Key, Value = x.Sum(y => y.AmountSubmitted) })
                .ToList();

            foreach (var item in claimsByMonth)
            {
                var date = new DateTime(item.Key.Year, item.Key.Month, 1);

                if (!result.ClaimsByMonth.ContainsKey(date))
                    result.ClaimsByMonth[date] = new List<Dtos.AdministratorDashboard.ClaimStatusValue>();

                result.ClaimsByMonth[date].Add(new Dtos.AdministratorDashboard.ClaimStatusValue { Status = item.Key.Status, Value = item.Value ?? 0 });
            }

            return result;
        }, true);

        return dashboard;
    }

    public async Task<Dtos.CustomerDashboard> Get(Customer customer)
    {
        var dashboard = await _cacheService.Get($"Dashboard-Customer-{customer.ID}", 300, () =>
        {
            var result = new Dtos.CustomerDashboard();

            result.LifetimeSavings = new Dtos.DashboardAggregate(488208.90m, 470019.88m, Dtos.DashboardAggregate.Type.Money, Dtos.DashboardAggregate.Period.Month);
            result.RecoveryRate = new Dtos.DashboardAggregate(47.2m, 44.9m, Dtos.DashboardAggregate.Type.Percentage, Dtos.DashboardAggregate.Period.Month);
            result.ClaimsValueUnderInvestigation = new Dtos.DashboardAggregate(109404.75m, 138881.62m, Dtos.DashboardAggregate.Type.Money, Dtos.DashboardAggregate.Period.Month);
            result.NewOrders = new Dtos.DashboardAggregate(3, 2, Dtos.DashboardAggregate.Type.Integer, Dtos.DashboardAggregate.Period.Month);
           
            result.ClaimsByState = _db.Claims
                .Include(x => x.Policy.State)
                .Where(x => x.Policy.CustomerID == customer.ID)
                .Where(x => x.Status != ClaimStatus.Tombstoned)
                .GroupBy(x => x.Policy.State.Code)
                .ToDictionary(x => x.Key, x => x.Count());

            result.ClaimsByMonth = new Dictionary<DateTime, List<Dtos.CustomerDashboard.ClaimStatusValue>>();

            var yearAgo = new DateOnly(DateTime.Now.Year, DateTime.Now.Month, 1).AddMonths(-12);
            var claimsByMonth = _db.Claims
                .Where(x => x.Policy.CustomerID == customer.ID)
                .Where(x => x.Status != ClaimStatus.Tombstoned)
                .Where(x => x.EventDate >= yearAgo)
                .GroupBy(x => new { x.Status, x.EventDate.Month, x.EventDate.Year })
                .Select(x => new { x.Key, Value = x.Sum(y => y.AmountSubmitted) })
                .ToList();

            foreach (var item in claimsByMonth)
            {
                var date = new DateTime(item.Key.Year, item.Key.Month, 1);

                if (!result.ClaimsByMonth.ContainsKey(date))
                    result.ClaimsByMonth[date] = new List<Dtos.CustomerDashboard.ClaimStatusValue>();

                result.ClaimsByMonth[date].Add(new Dtos.CustomerDashboard.ClaimStatusValue { Status = item.Key.Status, Value = item.Value ?? 0 });
            }

            return result;
        }, true);

        return dashboard;
    }

    public async Task<Dtos.InvestigatorDashboard> Get(Investigator investigator)
    {
        var dashboard = await _cacheService.Get($"Dashboard-Investigator-{investigator.ID}", 300, () =>
        {
            var result = new Dtos.InvestigatorDashboard();

            result.LifetimeEarnings = new Dtos.DashboardAggregate(271709.80m, 250110.79m, Dtos.DashboardAggregate.Type.Money, Dtos.DashboardAggregate.Period.Month);
            result.RecoveryRate = new Dtos.DashboardAggregate(59.0m, 61.9m, Dtos.DashboardAggregate.Type.Percentage, Dtos.DashboardAggregate.Period.Month);
            result.ClaimsValueUnderInvestigation = new Dtos.DashboardAggregate(34989.10m, 42911.31m, Dtos.DashboardAggregate.Type.Money, Dtos.DashboardAggregate.Period.Month);
            result.NewOrders = new Dtos.DashboardAggregate(1, 0, Dtos.DashboardAggregate.Type.Integer, Dtos.DashboardAggregate.Period.Month);

            result.ClaimsByState = _db.Claims
                .Include(x => x.Policy.State)
                .Where(x => x.InvestigatorID == investigator.ID)
                .Where(x => x.Status != ClaimStatus.Tombstoned)
                .GroupBy(x => x.Policy.State.Code)
                .ToDictionary(x => x.Key, x => x.Count());

            result.ClaimsByMonth = new Dictionary<DateTime, List<Dtos.InvestigatorDashboard.ClaimStatusValue>>();

            var yearAgo = new DateOnly(DateTime.Now.Year, DateTime.Now.Month, 1).AddMonths(-12);
            var claimsByMonth = _db.Claims
                .Where(x => x.InvestigatorID == investigator.ID)
                .Where(x => x.Status != ClaimStatus.Tombstoned)
                .Where(x => x.EventDate >= yearAgo)
                .GroupBy(x => new { x.Status, x.EventDate.Month, x.EventDate.Year })
                .Select(x => new { x.Key, Value = x.Sum(y => y.AmountSubmitted) })
                .ToList();

            foreach (var item in claimsByMonth)
            {
                var date = new DateTime(item.Key.Year, item.Key.Month, 1);

                if (!result.ClaimsByMonth.ContainsKey(date))
                    result.ClaimsByMonth[date] = new List<Dtos.InvestigatorDashboard.ClaimStatusValue>();

                result.ClaimsByMonth[date].Add(new Dtos.InvestigatorDashboard.ClaimStatusValue { Status = item.Key.Status, Value = item.Value ?? 0 });
            }

            return result;
        }, true);

        return dashboard;
    }
}
