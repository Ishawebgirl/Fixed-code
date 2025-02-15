using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure.Internal;
using Reclaim.Api.Model;
using System.ComponentModel.DataAnnotations;
using System.Data;

namespace Reclaim.Api.Services;

public class InvestigatorService : BaseService
{
    private readonly DatabaseContext _db;
    private readonly IHttpContextAccessor _httpContextAccessor;
    private readonly EmailService _emailService;
    private readonly SecurityService _securityService;
    private readonly MasterDataService _masterDataService;
    
    public InvestigatorService(DatabaseContext db, IHttpContextAccessor httpContextAccessor, EmailService emailService, SecurityService securityService, MasterDataService masterDataService)
    {
        _db = db;
        _httpContextAccessor = httpContextAccessor;
        _emailService = emailService;
        _securityService = securityService;
        _masterDataService = masterDataService;
    }

    private IQueryable<Investigator> GetQuery()
    {
        var (accountID, role) = base.GetCurrentAccountInfo(_httpContextAccessor);

        IQueryable<Investigator> query = _db.Investigators
            .Include(x => x.Account)
            .Include(x => x.State);
        
        switch (role)
        {
            case Role.Administrator:
                break;

            case Role.Customer:
                query = query.Where(x => x.Claims
                    .Any(y => y.Policy.Customer.AccountID == accountID));
                break;

            case Role.Investigator:
                query = query.Where(x => x.AccountID == accountID);
                break;
        }

        return query;
    }

    public async Task<Investigator> Get()
    {
        return await GetQuery()
            .FirstOrDefaultAsync();
    }

    public async Task<Investigator> Get(Guid uniqueID)
    {
        var investigator = await GetQuery()
            .FirstOrDefaultAsync(x => x.UniqueID == uniqueID);

        if (investigator == null)
            throw new ApiException(ErrorCode.InvestigatorDoesNotExistForAccount, $"Investigator {uniqueID} does not exist or is not associated with the current account");

        return investigator;
    }

    public async Task<List<Investigator>> GetAll()
    {
        var investigators = await GetQuery()
            .ToListAsync();

        return investigators;
    }


    public async Task<Investigator> Get(Account account)
    {
        var customer = await GetQuery().FirstOrDefaultAsync();

        return customer;
    }

    public async Task<Investigator> Create(Dtos.InvestigatorCreateOrUpdate dto, IdentityProvider provider, string? password = null)
    {
        var error = "Failed to create investigator account";
        var validationResults = new List<ValidationResult>();

        if (!Validator.TryValidateObject(dto, new ValidationContext(dto), validationResults))
            throw new ApiException(ErrorCode.ModelValidationFailed, $"{error}, ", validationResults);

        var state = await _masterDataService.GetStateByAbbreviation(dto.State);
        if (state == null) throw new ApiException(ErrorCode.MasterDataValueDoesNotExist, $"{error}, a state with abbreviation {dto.State} does not exist in the system.", "State");

        var account = await _securityService.Build(dto.EmailAddress, password, Role.Investigator, provider, $"{dto.FirstName} {dto.LastName}");
        var investigator = new Investigator
        {
            Account = account,
            Status = InvestigatorStatus.OnProbation,
            FirstName = dto.FirstName,
            LastName = dto.LastName,
            Address = dto.Address.ToNameCase(),
            Address2 = dto.Address2?.ToNameCase(),
            City = dto.City.ToNameCase(),
            StateID = state.ID,
            PostalCode = dto.PostalCode,
            Telephone = dto.Telephone
        };

        _db.Accounts.Add(account);
        _db.Investigators.Add(investigator);

        await _db.SaveChangesAsync();

        await _emailService.SendAccountConfirmation(investigator.Account, investigator.FirstName, password != null);

        return await Get(investigator.UniqueID);
    }

    public async Task<Investigator> Update(Guid uniqueID, Dtos.InvestigatorCreateOrUpdate dto)
    {
        var error = "Failed to update investigator account";
        var investigator = await Get(uniqueID);

        if (investigator == null)
            throw new ApiException(ErrorCode.InvestigatorDoesNotExist, $"{error}, investigator {uniqueID} does not exist.");

        var validationResults = new List<ValidationResult>();

        if (!Validator.TryValidateObject(dto, new ValidationContext(dto), validationResults))
            throw new ApiException(ErrorCode.ModelValidationFailed, $"{error}, ", validationResults);

        var state = await _masterDataService.GetStateByAbbreviation(dto.State);
        if (state == null) throw new ApiException(ErrorCode.MasterDataValueDoesNotExist, $"{error}, a state with abbreviation {dto.State} does not exist in the system.", "State");

        await _securityService.ValidateEmailAddressUpdate(dto.EmailAddress, investigator.Account, error);

        investigator.FirstName = dto.FirstName;
        investigator.LastName = dto.LastName;
        investigator.Address = dto.Address.ToNameCase();
        investigator.Address2 = dto.Address2?.ToNameCase();
        investigator.City = dto.City.ToNameCase();
        investigator.StateID = state.ID;
        investigator.PostalCode = dto.PostalCode;
        investigator.Telephone = dto.Telephone;
        investigator.Account.EmailAddress = dto.EmailAddress.ToLower();
        investigator.Account.NiceName = $"{dto.FirstName} {dto.LastName}";

        await _db.SaveChangesAsync();

        return await Get(uniqueID);
    }
}