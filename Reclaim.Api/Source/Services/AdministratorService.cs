using Microsoft.EntityFrameworkCore;
using Reclaim.Api.Model;
using System.ComponentModel.DataAnnotations;
using System.Data;

namespace Reclaim.Api.Services;

public class AdministratorService
{
    private readonly DatabaseContext _db;
    private readonly IHttpContextAccessor _httpContextAccessor;

    public AdministratorService(DatabaseContext db, IHttpContextAccessor httpContextAccessor)
    {
        _db = db;
        _httpContextAccessor = httpContextAccessor;
    }

    public async Task<List<Administrator>> GetAll()
    {
        var administrators = await _db.Administrators
            .Include(x => x.Account)
            .OrderBy(x => x.LastName)
            .ToListAsync();

        return administrators;
    }
}
