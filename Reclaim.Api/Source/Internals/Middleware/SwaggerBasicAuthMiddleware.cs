using System.Net;

namespace Reclaim.Api;

public class SwaggerBasicAuthMiddleware
{
    private readonly RequestDelegate next;
    private readonly string[] swaggerPaths = ["/swagger/v1/swagger.json", "/index.html"];
    private readonly string[] excludedHosts = ["localhost", "10.211.55.3:50000"];
    public SwaggerBasicAuthMiddleware(RequestDelegate next)
    {
        this.next = next;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        if (swaggerPaths.Contains(context.Request.Path.Value) && !excludedHosts.Contains(context.Request.Headers.Host.ToString()))
        {
            string authHeader = context.Request.Headers["swagger-ui-token"];

            if (authHeader != null && authHeader == Setting.SwaggerUiToken)
            {
                await next.Invoke(context).ConfigureAwait(false);
                return;
            }
            else
                context.Response.StatusCode = (int)HttpStatusCode.NotFound;
        }
        else
        {
            await next.Invoke(context).ConfigureAwait(false);
        }
    }
}


public static class AuthorizedSwaggerExtention
{
    public static IApplicationBuilder UseSwaggerAuthorized(this IApplicationBuilder builder)
    {
        return builder.UseMiddleware<SwaggerBasicAuthMiddleware>();
    }
}