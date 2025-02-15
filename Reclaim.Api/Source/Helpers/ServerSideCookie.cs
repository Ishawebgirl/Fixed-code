namespace Reclaim.Api
{
    public static class ServerSideCookie
    {
        public static string GetValue(HttpContext context, string key)
        {
            var cookies = context.Request.Headers.Where(H => H.Key.ToLower() == "cookie");

            if (cookies == null)
                return null;

            var cookieArray = cookies.FirstOrDefault().Value;

            if (cookieArray.Count() == 0)
                return null;

            var cookiePair = cookieArray.FirstOrDefault().Split(new char[] { ';' }).Where(C => C.Trim().StartsWith(key)).FirstOrDefault();

            if (cookiePair == null)
                return null;

            var cookie = cookiePair.Trim().Substring(key.Length + 1);
            return cookie;
        }
    }
}
