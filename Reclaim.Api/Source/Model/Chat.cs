using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class Chat : Base
{
    public int ID { get; set; }

    public ChatType Type { get; set; }

    public int AccountID { get; set; }

    public int ClaimID { get; set; }

    public Guid UniqueID { get; set; }

    public DateTime StartedTimestamp { get; set; }



    public virtual Account Account { get; set; } = null!;

    public virtual ICollection<ChatMessage> Messages { get; set; } = new List<ChatMessage>();

    public virtual Claim Claim { get; set; } = null!;
}
