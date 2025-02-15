using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class JobEvent : Base
{
    public int ID { get; set; }

    public int JobID { get; set; }

    public int? LogEntryID { get; set; }

    public int? ItemCount { get; set; }

    public string? Message { get; set; }

    public DateTime? StartedTimestamp { get; set; }

    public DateTime? FinishedTimestamp { get; set; }

    public DateTime? TimedOutTimestamp { get; set; }



    public virtual Job Job { get; set; } = null!;

    public virtual LogEntry? LogEntry { get; set; }
}
