using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class State : Base
{
    public int ID { get; set; }

    public string Code { get; set; } = null!;

    public string Name { get; set; } = null!;

    public int Sequence { get; set; }

    public bool IsEnabled { get; set; }



    public virtual ICollection<Customer> Customers { get; set; } = new List<Customer>();

    public virtual ICollection<Investigator> Investigators { get; set; } = new List<Investigator>();

    public virtual ICollection<Policy> Policies { get; set; } = new List<Policy>();
}
