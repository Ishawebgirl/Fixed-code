using Reclaim.Api.Model;
using System.Transactions;

namespace Reclaim.Api.Services;

public class LogService
{
    private readonly IHttpContextAccessor _httpContextAccessor;

    public LogService(IHttpContextAccessor httpContextAccessor)
    {
        _httpContextAccessor = httpContextAccessor;
    }

    public async Task<LogEntry> Add(string text)
    {
        var entry = new LogEntry { Text = text, Level = LogEntryLevel.Default };

        return await Add(entry);
    }

    public async Task<LogEntry> Add(LogEntryLevel level, string text, params object[] paramList)
    {
        var entry = new LogEntry { Text = string.Format(text, paramList), Level = level };

        return await Add(entry);
    }

    public async Task<LogEntry> Add(ApiException ex, string stackTrace)
    {
        var entry = new LogEntry { Text = $"{ex.DeepMessage()}", Level = LogEntryLevel.Default, ErrorCode = ex.ErrorCode, StackTrace = stackTrace };

        return await Add(entry);
    }

    public async Task<LogEntry> Add(ApiException ex)
    {
        var entry = new LogEntry { Text = $"{ex.DeepMessage()}", Level = LogEntryLevel.Default, ErrorCode = ex.ErrorCode, StackTrace = ex.StackTrace != null ? ex.StackTraceEx() : null };

        return await Add(entry);
    }

    public async Task<LogEntry> Add(Exception ex)
    {
        if (ex is ApiException)
            return await Add((ApiException)ex);
        else
        {
            var entry = new LogEntry { Text = $"An unhandled exception occurred: {ex.DeepMessage()}", Level = LogEntryLevel.Default, ErrorCode = ErrorCode.Unhandled, StackTrace = ex.StackTraceEx() };

            return await Add(entry);
        }
    }

    public async Task<LogEntry> Add(ErrorCode errorCode, Exception ex)
    {
        if (ex is ApiException)
            return await Add((ApiException)ex);
        else
        {
            var entry = new LogEntry { Text = $"An unhandled exception occurred: {ex.DeepMessage()}", Level = LogEntryLevel.Default, ErrorCode = errorCode, StackTrace = ex.StackTraceEx() };

            return await Add(entry);
        }
    }

#pragma warning disable CS1998 // Async method lacks 'await' operators and will run synchronously
    private async Task<LogEntry?> Add(LogEntry entry)
#pragma warning restore CS1998 // Async method lacks 'await' operators and will run synchronously
    {
        if (entry.ErrorCode == ErrorCode.ApplicationSettingsInvalid)
            return null;

        if ((int)entry.Level > Setting.LogEntryLevel)
            return null;

        if (_httpContextAccessor.HttpContext?.Request?.Path != null)
            entry.Url = _httpContextAccessor.HttpContext.Request.Path;

        if (_httpContextAccessor.HttpContext != null)
        {
            var claims = _httpContextAccessor.HttpContext.User.Claims;
            var accountIDString = claims.Where(c => c.Type == "AccountID").FirstOrDefault();

            if (accountIDString != null)
            {
                var accountIDStringDecrypted = TwoWayEncryption.Decrypt(accountIDString.Value);
                var accountID = accountIDStringDecrypted.ToNullableInteger();

                if (accountID != null)
                    entry.AccountID = accountID;
            }
        }

        if (Setting.SaveStackTraceInLogEntry)
        {
            if (entry.StackTrace == null)
                entry.StackTrace = Environment.StackTrace;

            entry.StackTrace = SanitizeStackTrace(entry.StackTrace);
        }
        else
            entry.StackTrace = null;

        entry.GeneratedTimestamp = DateTime.UtcNow;

        using (var scope = new TransactionScope(TransactionScopeOption.Suppress))
        {
            using (var db = new Model.DatabaseContext(DbContextOptions.Get()))
            {
                db.LogEntries.Add(entry);
                db.SaveChanges();
            }
        }

        return entry;
    }

    private string SanitizeStackTrace(string stackTrace)
    {
        if (stackTrace == null)
            return null;

        stackTrace = stackTrace.Replace("   at ", "at ");

        var endOfLogging = stackTrace.LastIndexOf("Services.LogService");

        if (endOfLogging >= 0)
        {
            var beginningOfMeaningfulError = stackTrace.IndexOf("at ", endOfLogging);
            stackTrace = stackTrace.Substring(beginningOfMeaningfulError);
        }

        var beginningofAsync = stackTrace.IndexOf("\r\nat System.Runtime.CompilerServices.AsyncMethodBuilderCore.Start");

        if (beginningofAsync >= 0)
            stackTrace = stackTrace.Left(beginningofAsync);

        var beginningOfRunInternal = stackTrace.IndexOf("\r\nat System.Threading.ExecutionContext.RunInternal");

        if (beginningOfRunInternal >= 0)
            stackTrace = stackTrace.Left(beginningOfRunInternal);

        var beginningOfMvc = stackTrace.IndexOf("at Microsoft.AspNetCore.Mvc.Infrastructure.ActionMethodExecutor");

        if (beginningOfMvc >= 0)
            stackTrace = stackTrace.Left(beginningOfMvc);

        return stackTrace;
    }
}