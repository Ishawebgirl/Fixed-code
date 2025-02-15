using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Reclaim.Api.Model;
using Reclaim.Api.Services;

namespace Reclaim.Api.Controllers;

[RequireRole]
[ValidateModel]
public class CustomerController : BaseController
{
    private readonly CustomerService _customerService;
    private readonly ClaimService _claimService;

    public CustomerController(CustomerService customerService, ClaimService claimService)
    {
        _customerService = customerService;
        _claimService = claimService;
    }

    /// <summary>
    /// Retrieve all customers for the current account's role, currently not paged or limited
    /// </summary>
    [HttpGet("customers")]
    [RequireRole]
    public async Task<List<Dtos.Customer>> GetCustomers()
    {
        var customers = await _customerService.GetAll();

        return customers.Select(x => new Dtos.Customer(x)).ToList();
    }

    /// <summary>
    /// Create a new customer
    /// </summary>
    /// <remarks>
    /// ErrorCode.ModelValidationFailed
    /// ErrorCode.MasterDataValueDoesNotExist
    /// ErrorCode.CustomerCodeAlreadyExists
    /// ErrorCode.AccountPasswordDoesNotMeetMinimumComplexity
    /// ErrorCode.
    /// </remarks>
    /// <param name="dto" example="">A CustomerCreateOrUpdate DTO</param>   
    [HttpPost("customers")]
    [RequireRole([Role.Administrator])]
    public async Task<Dtos.Customer> CreateCustomer([FromBody] Dtos.CustomerCreateOrUpdate dto)
    {
        var customer = await _customerService.Create(dto, IdentityProvider.Local);

        return new Dtos.Customer(customer);
    }

    /// <summary>
    /// Create a new customer, via the self-service registration workflow
    /// </summary>
    /// <remarks>
    /// ErrorCode.CustomerDoesNotExist
    /// ErrorCode.RequiredParameterNullOrEmpty
    /// ErrorCode.MasterDataValueDoesNotExist
    /// ErrorCode.ParameterCouldNotBeParsed
    /// ErrorCode.AccountEmailAddressInvalid
    /// ErrorCode.AccountPasswordDoesNotMeetMinimumComplexity
    /// </remarks>
    /// <param name="dto" example="">A CustomerRegistration DTO</param>
    [HttpPost("customers/registration")]
    [AllowAnonymous]
    public async Task<Dtos.Customer> Register([FromBody] Dtos.CustomerRegistration dto)
    {
        var customer = null as Customer;

        try
        {
            if (!string.IsNullOrEmpty(dto.GoogleCredential))
                customer = await _customerService.RegisterExternal(IdentityProvider.Google, dto);
            else
                customer = await _customerService.Register(dto);
        }
        catch (ApiException ex)
        {
            // hide informative messages to avoid dictionary attacks
            switch (ex.ErrorCode)
            {
                case ErrorCode.RequiredParameterNullOrEmpty:
                case ErrorCode.MasterDataValueDoesNotExist:
                case ErrorCode.ParameterCouldNotBeParsed:
                case ErrorCode.AccountEmailAddressInvalid:
                case ErrorCode.AccountPasswordDoesNotMeetMinimumComplexity:
                    throw;
            }

            return null;
        }

        return new Dtos.Customer(customer);
    }

    /// <summary>
    /// Retrieve a particular customer in the system
    /// </summary>
    /// <remarks>
    /// ErrorCode.CustomerDoesNotExist
    /// </remarks>e
    [HttpGet("customers/{uniqueID}")]
    [RequireRole]
    public async Task<Dtos.Customer> GetCustomer(Guid uniqueID)
    {
        var customer = await _customerService.Get(uniqueID);

        return new Dtos.Customer(customer);
    }

    /// <summary>
    /// Update an existing customer
    /// </summary>
    /// <remarks>
    /// ErrorCode.CustomerDoesNotExist
    /// ErrorCode.ModelValidationFailed
    /// ErrorCode.MasterDataValueDoesNotExist
    /// ErrorCode.CustomerCodeAlreadyExists
    /// ErrorCode.AccountPasswordDoesNotMeetMinimumComplexity
    /// ErrorCode.AccountEmailAddressAlreadyExists
    /// </remarks>
    /// <param name="uniqueID">The customer's unique public ID</param>
    /// <param name="dto" example="">A CustomerCreateOrUpdate DTO</param>
    [HttpPut("customers/{uniqueID}")]
    [RequireRole([Role.Administrator])]
    public async Task<Dtos.Customer> UpdateCustomer([FromRoute] Guid uniqueID, [FromBody] Dtos.CustomerCreateOrUpdate dto)
    {
        var customer = await _customerService.Update(uniqueID, dto);

        return new Dtos.Customer(customer);
    }

    /// <summary>
    /// Retrieve all claims for the given customer
    /// </summary>
    /// <remarks>
    /// ErrorCode.CustomerDoesNotExist
    /// </remarks>
    [HttpGet("customers/{uniqueID}/claims")]
    [RequireRole([Role.Administrator])]
    public async Task<List<Dtos.Claim>> GetClaimsByCustomer(Guid uniqueID)
    {
        var customer = await _customerService.Get(uniqueID);
        var claims = await _claimService.GetAll(customer);

        return claims.Select(x => new Dtos.Claim(x)).ToList();
    }
}
