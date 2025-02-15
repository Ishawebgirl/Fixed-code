using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class LogEntry : Base
{
    public int ID { get; set; }

    public LogEntryLevel Level { get; set; }

    public int? AccountID { get; set; }

    public ErrorCode? ErrorCode { get; set; }

    public DateTime GeneratedTimestamp { get; set; }

    public string? Url { get; set; }

    public string Text { get; set; } = null!;

    public string? StackTrace { get; set; }



    public virtual Account? Account { get; set; }

    public virtual ICollection<JobEvent> JobEvents { get; set; } = new List<JobEvent>();
}
