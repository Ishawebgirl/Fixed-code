using System.ComponentModel.DataAnnotations;
using System.Diagnostics.CodeAnalysis;

namespace Reclaim.Api.Dtos;

public record Job : Base
{
    /// <example>1</example>
    [Required]
    public int ID { get; set; }

    /// <example>RunDiagnostics</example>
    [Required]
    public Model.JobType Type { get; set; }

    /// <example>Pending</example>
    [Required]
    public Model.JobStatus Status { get; set; }

    /// <example>RunDiagnostics</example>
    [Required]
    public string Name { get; set; }

    /// <example>Run diagnostics</example>
    [Required]
    public string Description { get; set; }

    /// <example>86400</example>
    [Required]
    public int Interval { get; set; }

    /// <example>30</example>
    [Required]
    public int Timeout { get; set; }


    public DateTime? NextEvent { get; set; }

    [SetsRequiredMembers]
    public Job(Model.Job model)
    {
        ID = model.ID;
        Type = model.Type;
        Status = model.Status;
        Name = model.Name;
        Description = model.Description;
        Interval = model.Interval;
        Timeout = model.Timeout;
        NextEvent = model.NextEvent;
    }
}
