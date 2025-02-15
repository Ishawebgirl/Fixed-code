using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class EmailTemplate : Base
{
    public int ID { get; set; }

    public string Code { get; set; } = null!;

    public string ExternalID { get; set; } = null!;

    public string Subject { get; set; } = null!;

    public string Preheader { get; set; } = null!;

    public string Body { get; set; } = null!;

    public string HighlightColor { get; set; } = null!;

    public string? ActionButtonText { get; set; }



    public virtual ICollection<Email> Emails { get; set; } = new List<Email>();
}
