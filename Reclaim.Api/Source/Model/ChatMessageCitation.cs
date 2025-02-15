using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class ChatMessageCitation : Base
{
    public int ID { get; set; }

    public int ChatMessageID { get; set; }

    public int DocumentID { get; set; }

    public int PageNumber { get; set; }

    public string BoundingBoxes { get; set; } = null!;

    public Guid UniqueID { get; set; }



    public virtual ChatMessage ChatMessage { get; set; } = null!;

    public virtual Document Document { get; set; } = null!;
}
