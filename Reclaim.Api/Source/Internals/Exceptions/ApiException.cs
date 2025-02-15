using Reclaim.Api.Model;
using System.ComponentModel.DataAnnotations;

namespace Reclaim.Api;

public class ApiException : Exception
{
    public ErrorCode ErrorCode { get; set; }
    public int LogEntryID { get; set; }
    public LogEntry LogEntry { get; set; }
    public List<string> Fields { get; set; }

    public ApiException(ErrorCode errorCode) : base("")
    {
        ErrorCode = errorCode;
    }

    public ApiException(ErrorCode errorCode, string message, List<ValidationResult> results) : this(errorCode, string.Join(", ", results.Select(x => x.ErrorMessage)))
    {
        ErrorCode = errorCode;
    }

    public ApiException(ErrorCode errorCode, string message) : base(message)
    {
        ErrorCode = errorCode;
    }
    public ApiException(ErrorCode errorCode, string message, string field) : base(message)
    {
        ErrorCode = errorCode;
        Fields = new List<string> { field };
    }

    public ApiException(ErrorCode errorCode, string message, List<string> fields) : base(message)
    {
        ErrorCode = errorCode;
        Fields = fields;
    }

    public ApiException(Exception innerException, ErrorCode errorCode) : base(innerException.Message, innerException)
    {
        ErrorCode = errorCode;
    }

    public ApiException(Exception innerException, ErrorCode errorCode, string message) : base(message, innerException)
    {
        ErrorCode = errorCode;
    }
}