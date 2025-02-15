namespace Reclaim.Api;

public class ReplaceBearerTokenWithCookie
{
    private readonly RequestDelegate _next;

    public ReplaceBearerTokenWithCookie(RequestDelegate next)
    {
        _next = next;
    }

    public async Task Invoke(HttpContext context)
    {
        var accessToken = ServerSideCookie.GetValue(context, CookieName.AccessToken);
        var authKey = "Authorization";

        if (!string.IsNullOrEmpty(accessToken))
        {
            if (context.Request.Headers.ContainsKey(authKey))
                context.Request.Headers.Remove(authKey);

            context.Request.Headers.Append(authKey, new string[] { $"Bearer {accessToken}" });
        }

        await _next(context);
    }
}

