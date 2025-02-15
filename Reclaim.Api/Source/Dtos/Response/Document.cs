using Reclaim.Api.Model;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics.CodeAnalysis;

namespace Reclaim.Api.Dtos;

public record Document
{
    /// <example>024528fa-e5bc-45c1-b7aa-06c03956adf3</example>
    [Required]
    public Guid UniqueID { get; set; }

    /// <example>Mp4</example>
    [Required]
    public DocumentType Type { get; set; }

    /// <example>Plat survey 2024-10-06.pdf</example>
    [Required]
    public string Name { get; set; }

    /// <example>/HARPETH/Plat survey 2024-10-06.pdf</example>
    [Required]
    public string Path { get; set; }

    /// <example>8837107</example>
    [Required]
    public int Size { get; set; }

    /// <example>B2D9E238940E8A459ED485C250B1FB0B68F90A0F</example>
    [Required]
    public string Hash { get; set; }

    /// <example>General overview of fire damage to exterior structure</example>
    [Required]
    public string Description { get; set; }

    /// <example>null</example>
    public string? Summary { get; set; }

    public DateTime? OriginatedTimestamp { get; set; }
    
    [Required]
    public DateTime UploadedTimestamp { get; set; }
    
    public DateTime? SummarizedTimestamp { get; set; }
    public DateTime? TombstonedTimestamp { get; set; }

    [SetsRequiredMembers]
    public Document(Model.Document model)
    {
        UniqueID = model.UniqueID;
        Type = model.Type;
        Name = model.Name;
        Path = model.Path;
        Size = model.Size;
        Hash = model.Hash;
        Description = model.Description;
        Summary = model.Summary;
        OriginatedTimestamp = model.OriginatedTimestamp;
        UploadedTimestamp = model.UploadedTimestamp;
        SummarizedTimestamp = model.SummarizedTimestamp;
        TombstonedTimestamp = model.TombstonedTimestamp;
    }
}
