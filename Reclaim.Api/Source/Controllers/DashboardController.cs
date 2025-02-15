using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Reclaim.Api.Model;
using Reclaim.Api.Services;
using Xunit.Sdk;

namespace Reclaim.Api.Controllers;

[RequireRole]
[ValidateModel]
public class DashboardController : BaseController
{
    private readonly DashboardService _dashboardService;
    private readonly CustomerService _customerService;
    private readonly InvestigatorService _investigatorService;

    public DashboardController(DashboardService dashboardService, CustomerService customerService, InvestigatorService investigatorService)
    {
        _dashboardService = dashboardService;
        _customerService = customerService;
        _investigatorService = investigatorService;
    }

    /// <summary>
    /// Retrieve an object containing aggregate values to populate the administrator's landing page
    /// </summary>
    [HttpGet("dashboard/administrator")]
    [RequireRole(Role.Administrator)]
    public async Task<Dtos.AdministratorDashboard> GetAdministrator()
    {
        var dashboard = await _dashboardService.Get();

        return dashboard;
    }

    /// <summary>
    /// Retrieve an object containing aggregate values to populate the administrator's landing page
    /// </summary>
    [HttpGet("dashboard/customer")]
    [RequireRole(Role.Customer)]
    public async Task<Dtos.CustomerDashboard> GetCustomer()
    {
        var customer = await _customerService.Get();
        var dashboard = await _dashboardService.Get(customer);

        return dashboard;
    }


    /// <summary>
    /// Retrieve an object containing aggregate values to populate the administrator's landing page
    /// </summary>
    [HttpGet("dashboard/investigator")]
    [RequireRole(Role.Investigator)]
    public async Task<Dtos.InvestigatorDashboard> GetInvestigator()
    {

        var investigator = await _investigatorService.Get();
        var dashboard = await _dashboardService.Get(investigator);

        return dashboard;
    }
}