using System.ComponentModel.DataAnnotations;
using System.Diagnostics.CodeAnalysis;

namespace Reclaim.Api.Dtos;

public record DashboardAggregate : Base
{
    public DashboardAggregate(decimal currentValue, decimal? previousValue, Type type, Period period)
    {
        CurrentValue = currentValue;    
        PreviousValue = previousValue;  
        ValueType = type;
        ComparisonPeriod = period;
    }

    public enum Type
    {
        Integer,
        Money,
        Percentage
    }

    public enum Period
    {
        Hour,
        Day,
        Week,
        Month,
        Year
    }

    /// <example>7811.15</example>
    [Required]
    public decimal CurrentValue { get; set; }

    /// <example>5808.90</example>
    [Required]
    public decimal? PreviousValue { get; set; }

    /// <example>Money</example>
    [Required]
    public Type ValueType { get; set; }

    /// <example>Month</example>
    [Required]
    public Period ComparisonPeriod { get; set; }

    [Required]
    public decimal? PercentChange
    {
        get
        {
            if (PreviousValue == null)
                return null;

            if (PreviousValue == 0)
                return null;

            return Math.Round((decimal)(100.0m * ((CurrentValue - PreviousValue) / PreviousValue)), 2);
        }
    }
}
