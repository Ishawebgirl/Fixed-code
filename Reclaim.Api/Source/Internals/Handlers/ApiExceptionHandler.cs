using Reclaim.Api.Model;
using Reclaim.Api.Services;
using System.Net;

namespace Reclaim.Api;

public class ApiExceptionHandler
{
    private readonly RequestDelegate _next;
    private EmailService _emailService;
    private LogService _logService;

    public ApiExceptionHandler(RequestDelegate next)
    {
        _next = next;
    }

    public async Task Invoke(HttpContext context, LogService logService, EmailService emailService)
    {
        _emailService = emailService;
        _logService = logService;

        try
        {
            await _next(context);
        }
        catch (Exception exception)
        {
            var response = context.Response;
            response.ContentType = "application/json";

            var content = "";
            var statusCode = HttpStatusCode.BadRequest;
            var ex = null as ApiException;

            if (exception is ApiException)
            {
                ex = exception as ApiException;

                var logEntry = await _logService.Add(ex);

                if (logEntry != null)
                    ex.LogEntryID = logEntry.ID;

                switch (ex.ErrorCode)
                {
                    case ErrorCode.Unhandled:
                    case ErrorCode.Unknown:
                        statusCode = HttpStatusCode.InternalServerError;
                        break;

                    case ErrorCode.AccountCredentialsInvalid:
                    case ErrorCode.AccountExternalCredentialsInvalid:
                    case ErrorCode.AccountEmailAddressNotConfirmed:
                    case ErrorCode.AccountTombstoned:
                    case ErrorCode.AccountLockedOut:
                    case ErrorCode.AccountLockedOutOverride:
                    case ErrorCode.GoogleJwtBearerTokenInvalid:
                    case ErrorCode.AccountCredentialsExpired:
                    case ErrorCode.JwtBearerTokenExpired:
                    case ErrorCode.JwtBearerTokenInvalid:
                    case ErrorCode.JwtBearerTokenMissing:
                    case ErrorCode.JwtRefreshTokenInvalid:
                    case ErrorCode.JwtClaimInvalid:
                    case ErrorCode.JwtClaimNotPresent:
                    case ErrorCode.JwtUnknownError:
                        statusCode = HttpStatusCode.Unauthorized;
                        break;

                    case ErrorCode.JwtRoleInvalid:
                        statusCode = HttpStatusCode.Forbidden;
                        break;

                    case ErrorCode.AccountRoleInvalidForOperation:
                        statusCode = HttpStatusCode.Forbidden;
                        break;

                    case ErrorCode.AccountDoesNotExist:
                        statusCode = HttpStatusCode.BadRequest;
                        break;

                    case ErrorCode.RateLimitExceeded:
                        statusCode = HttpStatusCode.TooManyRequests;
                        break;
                }
            }
            else
            {
                ex = new ApiException(ErrorCode.Unhandled, exception.Message);

                if (exception is Microsoft.EntityFrameworkCore.DbUpdateException)
                    ex = new ApiException(ErrorCode.EntityFrameworkError, exception.InnerException?.Message ?? exception.Message);
                else
                    ex = new ApiException(ErrorCode.Unhandled, exception.Message);

                var logEntry = await _logService.Add(ex.ErrorCode, ex);

                if (logEntry != null)
                    ex.LogEntryID = logEntry.ID;
            }

            if (ex.ErrorCode != ErrorCode.ApplicationSettingsInvalid && Setting.DeliverUnhandledExceptionEmails)
                try { await _emailService.SendUnhandledException(ex); } catch { }

            content = new Dtos.ApiException(ex).Serialize();
            response.StatusCode = (int)statusCode;

            await response.WriteAsync(content);
        }
    }

}
