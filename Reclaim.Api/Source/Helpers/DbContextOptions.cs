using Microsoft.EntityFrameworkCore;

namespace Reclaim.Api;

public static class DbContextOptions
{
    internal static DbContextOptions<Model.DatabaseContext> Get()
    {
        var options = new DbContextOptionsBuilder<Model.DatabaseContext>()
            .UseSqlServer($"{Environment.GetEnvironmentVariable("RECLAIM_API_CONNECTION_STRING")};Application Name=Reclaim.api")
            .Options;

        return options;
    }
}