using System.ComponentModel.DataAnnotations;
using System.Diagnostics.CodeAnalysis;

namespace Reclaim.Api.Dtos;

public record CustomerDashboard : Base
{
    public record ClaimStatusValue
    {
        public Model.ClaimStatus Status { get; set; }
        public decimal Value { get; set; }
    }

    public DashboardAggregate LifetimeSavings { get; set; }
    public DashboardAggregate RecoveryRate { get; set; }
    public DashboardAggregate NewOrders { get; set; }
    public DashboardAggregate ClaimsValueUnderInvestigation { get; set; }

    public Dictionary<string, int> ClaimsByState { get; set; }

    public Dictionary<DateTime, List<ClaimStatusValue>> ClaimsByMonth { get; set; }
}
