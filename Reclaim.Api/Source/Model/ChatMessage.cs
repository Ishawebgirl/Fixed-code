using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class ChatMessage : Base
{
    public int ID { get; set; }

    public int ChatID { get; set; }

    public ChatRole ChatRole { get; set; }

    public string Text { get; set; } = null!;

    public DateTime SubmittedTimestamp { get; set; }

    public DateTime? ReceivedTimestamp { get; set; }

    public string? Metadata { get; set; }

    public bool IsError { get; set; }

    public Guid UniqueID { get; set; }



    public virtual Chat Chat { get; set; } = null!;

    public virtual ICollection<ChatMessageCitation> Citations { get; set; } = new List<ChatMessageCitation>();
}
