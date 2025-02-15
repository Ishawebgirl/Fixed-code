using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Reclaim.Api.Model;
using Reclaim.Api.Services;

namespace Reclaim.Api.Controllers;

[RequireRole]
[ValidateModel]
public class InvestigatorController : BaseController
{
    private readonly InvestigatorService _investigatorService;
    private readonly ClaimService _claimService;

    public InvestigatorController(InvestigatorService investigatorService, ClaimService claimService)
    {
        _investigatorService = investigatorService;
        _claimService = claimService;
    }


    /// <summary>
    /// Retrieve all investigators associated with the role, currently not paged or limited
    /// </summary>
    [HttpGet("investigators")]
    [RequireRole]
    public async Task<List<Dtos.Investigator>> GetAll()
    {
        var investigators = await _investigatorService.GetAll();

        return investigators.Select(x => new Dtos.Investigator(x)).ToList();
    }

    /// <summary>
    /// Create a new investigator
    /// </summary>
    /// <remarks>
    /// ErrorCode.ModelValidationFailed
    /// ErrorCode.MasterDataValueDoesNotExist
    /// ErrorCode.AccountPasswordDoesNotMeetMinimumComplexity
    /// ErrorCode.AccountEmailAddressAlreadyExists
    /// </remarks>
    /// <param name="dto" example="">An InvestigatorCreateOrUpdate DTO</param>
    [HttpPost("investigators")]
    [RequireRole([Role.Administrator])]
    public async Task<Dtos.Investigator> Create([FromBody] Dtos.InvestigatorCreateOrUpdate dto)
    {
        var investigator = await _investigatorService.Create(dto, IdentityProvider.Local);

        return new Dtos.Investigator(investigator);
    }

    /// <summary>
    /// Retrieve a particular investigator in the system
    /// </summary>
    /// <remarks>
    /// ErrorCode.InvestigatorDoesNotExist
    /// </remarks>
    /// <param name="uniqueID">The investigator's unique public ID</param>
    [HttpGet("investigators/{uniqueID}")]
    [RequireRole]
    public async Task<Dtos.Investigator> Get(Guid uniqueID)
    {
        var investigator = await _investigatorService.Get(uniqueID);

        return new Dtos.Investigator(investigator);
    }

    /// <summary>
    /// Update an existing investigator
    /// </summary>
    /// <remarks>
    /// ErrorCode.InvestigatorDoesNotExist
    /// ErrorCode.ModelValidationFailed
    /// ErrorCode.MasterDataValueDoesNotExist
    /// ErrorCode.AccountPasswordDoesNotMeetMinimumComplexity
    /// ErrorCode.AccountEmailAddressAlreadyExists
    /// </remarks>
    /// <param name="uniqueID">The investigator's unique public ID</param>
    /// <param name="dto" example="">An Investigator DTO</param>
    [HttpPut("investigators/{uniqueID}")]
    [RequireRole([Role.Administrator])]
    public async Task<Dtos.Investigator> Update([FromRoute] Guid uniqueID, [FromBody] Dtos.InvestigatorCreateOrUpdate dto)
    {
        var investigator = await _investigatorService.Update(uniqueID, dto);

        return new Dtos.Investigator(investigator);
    }
}