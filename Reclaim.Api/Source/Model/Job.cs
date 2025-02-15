using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class Job : Base
{
    public int ID { get; set; }

    public JobType Type { get; set; }

    public JobStatus Status { get; set; }

    public string Name { get; set; } = null!;

    public string Description { get; set; } = null!;

    public int Interval { get; set; }

    public int Timeout { get; set; }

    public DateTime NextEvent { get; set; }



    public virtual ICollection<JobEvent> Events { get; set; } = new List<JobEvent>();
}
