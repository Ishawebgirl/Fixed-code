using System.ComponentModel.DataAnnotations;
using System.Diagnostics.CodeAnalysis;

namespace Reclaim.Api.Dtos;

public record ApiException : Base
{
    [Required]
    public int ErrorCode { get; set; }

    [Required]
    public string ErrorCodeName { get; set; }

    [Required]
    public string Message { get; set; }

    [Required]
    public int LogEntryID { get; set; }
    
    public List<string>? Fields { get; set; }

    public ApiException() { }

    [SetsRequiredMembers]
    public ApiException(Reclaim.Api.ApiException model)
    {
        ErrorCode = (int)model.ErrorCode;
        ErrorCodeName = model.ErrorCode.ToString();
        Message = model.Message;
        LogEntryID = model.LogEntryID;
        Fields = model.Fields;
    }
}