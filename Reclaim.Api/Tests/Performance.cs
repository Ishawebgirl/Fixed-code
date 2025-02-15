using Xunit;
using RestSharp;
using System.Threading;

namespace Reclaim.Api.Tests;

public class Performance : Base
{
    public Performance()
    {
        Database.Reset();
        
        _restClient = new RestClient(Default.Url);
    }

    [Fact(Skip = "To test rate limiting, make sure to remove the localhost hostname check in Program.cs")]
    public void RateLimitForAuthorization()
    {
        var warmup = GetJwtAccessToken();

        for (var i = 0; i < 4; i++)
        {
            var data = GetJwtAccessToken();
            data.statusCode.ShouldBe(200);
        }

        var final = GetJwtAccessToken();
        final.statusCode.ShouldBe(429);
        final.errorCodeName.ShouldBe("RateLimitExceeded");
        final.message.IndexOf("try again in 600s").ShouldNotBe(-1);
        Thread.Sleep(1000);
    }

    /*
    [Fact(Skip = "To test rate limiting, make sure to remove the localhost hostname check in Program.cs")]e
    public void RateLimitForOther()
    {
        var auth = GetJwtAccessToken();

        for (var i = 0; i < 30; i++)
        {
            var data = RequestWithJwtAccessToken("accounts/me", Method.Get, null, auth.accessToken);
            Assert.True(data.statusCode, Is.EqualTo(200));
        }

        var final = RequestWithJwtAccessToken("accounts/me", Method.Get, null, auth.accessToken);
        final.statusCode.ShouldBe(429);
        final.errorCodeName.ShouldBe("RateLimitExceeded");
        final.message.IndexOf("try again in 600s").ShouldNotBe(-1);        
    }
    */
}
