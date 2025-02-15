using Microsoft.EntityFrameworkCore;
using Reclaim.Api.Model;
using System.Data;

namespace Reclaim.Api.Services;

public class ClaimService : BaseService
{
    private readonly DatabaseContext _db;
    private readonly IHttpContextAccessor _httpContextAccessor;

    public ClaimService(DatabaseContext db, IHttpContextAccessor httpContextAccessor)
    {
        _db = db;
        _httpContextAccessor = httpContextAccessor;
    }

    private IQueryable<Claim> GetQuery()
    {
        var (accountID, role) = base.GetCurrentAccountInfo(_httpContextAccessor);

        IQueryable<Claim> query = _db.Claims
            .Include(x => x.Policy.State)
            .Include(x => x.Investigator.State)
            .Include(x => x.Investigator.Account)
            .Include(x => x.Documents)
            .Include(x => x.Policy.Customer.State)
            .Include(x => x.Policy.Customer.Account);

        switch (role)
        {
            case Role.Administrator:
                break;

            case Role.Customer:
                query = query.Where(x => x.Policy.Customer.AccountID == accountID);
                break;

            case Role.Investigator:
                query = query.Where(x => x.Investigator.AccountID == accountID);
                break;
        }

        return query;
    }

   
    public async Task<List<Claim>> GetAll()
    {
        var claims = await GetQuery().ToListAsync();

        return claims;
    }

    public async Task<Claim> Get(Guid uniqueID)
    {
        var claim = await GetQuery()
            .FirstOrDefaultAsync(x => x.UniqueID == uniqueID);

        if (claim == null)
            throw new ApiException(ErrorCode.ClaimDoesNotExistForAccount, $"Claim {uniqueID} does not exist or is not associated with the current account");

        return claim;
    }

    public async Task<List<Claim>> GetAll(Customer customer)
    {
        var claims = await GetQuery()
            .Where(x => x.Policy.Customer.ID == customer.ID)
            .ToListAsync();

        return claims;
    }
}
