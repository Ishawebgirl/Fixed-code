using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace Reclaim.Api.Controllers;

[RequireRole]
[ValidateModel]
public class ContentController : BaseController
{
    private readonly Services.EmailService _emailService;

    public ContentController(Services.EmailService emailService)
    {
        _emailService = emailService;
    }
    /// <summary>
    /// Receives a notification when an email is opened (and image downloads are enabled)
    /// </summary>
    /// <param name="uniqueID" example="CF1FE0FB-2294-46BC-915E-293B6A1BC5A2">A system-wide unique ID for the given email message</param>
    /// <remarks></remarks>
    [HttpGet("content/emails/{uniqueID}/received.png")]
    [AllowAnonymous]
    public async Task<FileContentResult> SetEmailReceived([FromRoute] Guid? uniqueID)
    {
        await _emailService.SetReceived(uniqueID.Value);

        var fileName = "Content/Images/pixel.png";
        var content = await System.IO.File.ReadAllBytesAsync(fileName);

        return File(content, "image/png", "pixel.png");
    }
}
