using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class Email : Base
{
    public int ID { get; set; }

    public EmailStatus Status { get; set; }

    public int TemplateID { get; set; }

    public int AccountID { get; set; }

    public DateTime? DeliverAfter { get; set; }

    public DateTime? DeliveredTimestamp { get; set; }

    public DateTime? ReceivedTimestamp { get; set; }

    public DateTime? FailedTimestamp { get; set; }

    public DateTime? TombstonedTimestamp { get; set; }

    public Guid UniqueID { get; set; }



    public virtual Account Account { get; set; } = null!;

    public virtual EmailTemplate Template { get; set; } = null!;
}
