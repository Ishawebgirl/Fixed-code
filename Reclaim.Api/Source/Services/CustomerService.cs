using Microsoft.EntityFrameworkCore;
using Reclaim.Api.Model;
using System.ComponentModel.DataAnnotations;
using System.Data;

namespace Reclaim.Api.Services;

public class CustomerService : BaseService
{
    private readonly DatabaseContext _db;
    private readonly IHttpContextAccessor _httpContextAccessor;
    private readonly EmailService _emailService;
    private readonly SecurityService _securityService;
    private readonly MasterDataService _masterDataService;

    public CustomerService(DatabaseContext db, IHttpContextAccessor httpContextAccessor, EmailService emailService, SecurityService securityService, MasterDataService masterDataService)
    {
        _db = db;
        _httpContextAccessor = httpContextAccessor;
        _emailService = emailService;
        _securityService = securityService;
        _masterDataService = masterDataService;
    }


    private IQueryable<Customer> GetQuery()
    {
        var (accountID, role) = GetCurrentAccountInfo(_httpContextAccessor);

        IQueryable<Customer> query = _db.Customers
            .Include(x => x.Account)
            .Include(x => x.State);

        switch (role)
        {
            case Role.Administrator:
                break;

            case Role.Customer:
                query = query.Where(x => x.AccountID == accountID);                    
                break;

            case Role.Investigator:
                query = query.Where(x => x.Policies
                    .SelectMany(y => y.Claims)
                    .Any(z => z.Investigator.AccountID == accountID));
                break;
        }

        return query;
    }

    public async Task<List<Customer>> GetAll()
    {
        var customers = await GetQuery()
            .ToListAsync();

        return customers;
    }

    public async Task<Customer> Get()
    {
        var customer = await GetQuery().FirstOrDefaultAsync();           

        return customer;
    }

    public async Task<Customer> Get(Guid uniqueID)
    {
        var customer = await GetQuery()
            .FirstOrDefaultAsync(x => x.UniqueID == uniqueID);

        if (customer == null)
            throw new ApiException(ErrorCode.CustomerDoesNotExistForAccount, $"Customer {uniqueID} does not exist or is not associated with the current account");

        return customer;
    }

    public async Task<Customer> Create(Dtos.CustomerCreateOrUpdate dto, IdentityProvider provider, string? password = null)
    {
        var error = "Failed to create customer account";
        var validationResults = new List<ValidationResult>();

        if (!Validator.TryValidateObject(dto, new ValidationContext(dto), validationResults))
            throw new ApiException(ErrorCode.ModelValidationFailed, $"{error}, ", validationResults);

        var state = await _masterDataService.GetStateByAbbreviation(dto.State);
        if (state == null) throw new ApiException(ErrorCode.MasterDataValueDoesNotExist, $"{error}, a state with abbreviation {dto.State} does not exist in the system.", "State");

        dto.Code = dto.Code.ToUpper();
        await ValidateCustomerCode(dto.Code, null, error);

        var account = await _securityService.Build(dto.EmailAddress, password, Role.Customer, provider, $"{dto.FirstName} {dto.LastName}");
        var customer = new Customer
        {
            Account = account,
            Status = CustomerStatus.Uncommitted,
            Name = dto.Name,
            FirstName = dto.FirstName,
            LastName = dto.LastName,
            Code = dto.Code,
            Address = dto.Address.ToNameCase(),
            Address2 = dto.Address2?.ToNameCase(),
            City = dto.City.ToNameCase(),
            StateID = state.ID,
            PostalCode = dto.PostalCode,
            Telephone = dto.Telephone
        };

        _db.Accounts.Add(account);
        _db.Customers.Add(customer);

        await _db.SaveChangesAsync();

        await _emailService.SendAccountConfirmation(customer.Account, customer.FirstName, password != null);

        return await Get(customer.UniqueID);
    }

    private async Task ValidateCustomerCode(string requestedCode, Customer? customer, string error)
    {
        var existingCustomer = await _db.Customers.FirstOrDefaultAsync(x => x.Code == requestedCode);

        if (existingCustomer == null)
            return;

        if (customer != null && customer.ID == existingCustomer.ID)
            return;

        throw new ApiException(ErrorCode.CustomerCodeAlreadyExists, $"{error}, a customer with code {requestedCode} already exists.", "Code");
    }

    public async Task<Customer> Update(Guid uniqueID, Dtos.CustomerCreateOrUpdate dto)
    {
        var error = "Failed to update customer account";
        var customer = await Get(uniqueID);

        if (customer == null)
            throw new ApiException(ErrorCode.CustomerDoesNotExist, $"{error}, customer {uniqueID} does not exist.");

        var validationResults = new List<ValidationResult>();

        if (!Validator.TryValidateObject(dto, new ValidationContext(dto), validationResults))
            throw new ApiException(ErrorCode.ModelValidationFailed, $"{error}, ", validationResults);

        var state = await _masterDataService.GetStateByAbbreviation(dto.State);
        if (state == null) throw new ApiException(ErrorCode.MasterDataValueDoesNotExist, $"{error}, a state with abbreviation {dto.State} does not exist in the system.", "State");

        dto.Code = dto.Code.ToUpper();
        await ValidateCustomerCode(dto.Code, customer, error);

        await _securityService.ValidateEmailAddressUpdate(dto.EmailAddress, customer.Account, error);

        customer.Name = dto.Name;
        customer.FirstName = dto.FirstName;
        customer.LastName = dto.LastName;
        customer.Code = dto.Code;
        customer.Address = dto.Address.ToNameCase();
        customer.Address2 = dto.Address2?.ToNameCase();
        customer.City = dto.City.ToNameCase();
        customer.StateID = state.ID;
        customer.PostalCode = dto.PostalCode;
        customer.Telephone = dto.Telephone;
        customer.Account.EmailAddress = dto.EmailAddress.ToLower();
        customer.Account.NiceName = $"{dto.FirstName} {dto.LastName}";

        await _db.SaveChangesAsync();

        return await Get(uniqueID);
    }

    private async Task<string> GenerateUniqueCode(string name)
    {
        const int minLength = 2;
        const int maxLength = 10;
        const int attempts = 20;

        var code = name.ToUpper().Split(' ').Aggregate("", (current, word) => current + word.Substring(0, 1));

        if (code.Length < minLength)
            code += new string('X', 2 - code.Length);

        if (code.Length > maxLength)
            code = code.Left(maxLength);

        for (var i = 0; i < attempts; i++)
        {
            var option = i == 0 ? code : $"{code}{i}";
            var existingCustomer = await _db.Customers.FirstOrDefaultAsync(x => x.Code == option);
            
            if (existingCustomer == null)
                return option;
        }
       
        throw new ApiException(ErrorCode.CustomerCodeGenerationFailed, $"Failed to generate a unique customer code for {name}.");
    }

    public async Task<Customer> Register(Dtos.CustomerRegistration dto)
    {
        var code = await GenerateUniqueCode(dto.Name);

        var customerDto = new Dtos.CustomerCreateOrUpdate
        {
            Name = dto.Name,
            FirstName = dto.FirstName,
            LastName = dto.LastName,
            Code = code,
            Address = dto.Address,
            Address2 = dto.Address2,
            City = dto.City,
            State = dto.State,
            PostalCode = dto.PostalCode,
            EmailAddress = dto.EmailAddress,
            Telephone = dto.Telephone,
        };

        return await Create(customerDto, IdentityProvider.Local, dto.Password);
    }

    public async Task<Customer> RegisterExternal(IdentityProvider provider, Dtos.CustomerRegistration dto)
    {
        _securityService.ValidateGoogleCredential(dto.GoogleCredential, dto.EmailAddress, dto.FirstName, dto.LastName);
        var dummyPassword = _securityService.GeneratePassword();
        var code = await GenerateUniqueCode(dto.Name);

        var customerDto = new Dtos.CustomerCreateOrUpdate
        {
            Name = dto.Name,
            Code = code,
            FirstName = dto.FirstName,
            LastName = dto.LastName,
            Address = dto.Address,
            Address2 = dto.Address2,
            City = dto.City,
            State = dto.State,
            PostalCode = dto.PostalCode,
            EmailAddress = dto.EmailAddress,
            Telephone = dto.Telephone,
        };

        return await Create(customerDto, provider, dummyPassword);
    }
}
