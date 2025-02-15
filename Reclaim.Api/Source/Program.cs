using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Mvc.Infrastructure;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.FileProviders;
using Microsoft.OpenApi.Models;
using Reclaim.Api;
using Reclaim.Api.Model;
using Reclaim.Api.Services;
using System.Reflection;
using System.Text.Json.Serialization;
using System.Threading.RateLimiting;

var builder = WebApplication.CreateBuilder(args);
var services = builder.Services;

services.AddControllers(o => {
    o.Conventions.Add(new ControllerDocumentationConvention());
    })    
    .AddJsonOptions(x =>
    {
        x.JsonSerializerOptions.Converters.Add(new JsonStringEnumConverter());
        x.JsonSerializerOptions.Converters.Add(new JsonDateTimeConverter());
    });


services.AddCors(policyBuilder =>
    policyBuilder.AddPolicy("Cors-Default", p => p
        .WithOrigins(new string[] { "http://localhost:3000", "http://localhost:3001", "https://web.reclaimsiu.com", "https://reclaimsiu.com" })
        .AllowAnyHeader()
        .AllowAnyMethod()
        .AllowCredentials())
);

services.AddDbContext<DatabaseContext>(options =>
{
    options.UseSqlServer($"{Environment.GetEnvironmentVariable(Constant.ConnectionString)}");
},
    ServiceLifetime.Scoped);

services.AddDistributedSqlServerCache(options =>
{
    options.ConnectionString = $"{Environment.GetEnvironmentVariable(Constant.ConnectionString)};Application Name=Reclaim.api distributed cache";
    options.SchemaName = "dbo";
    options.TableName = "Cache";
});

const int maxBodySize = 50 * 1024 * 1024;
services.Configure<IISServerOptions>(o => o.MaxRequestBodySize = maxBodySize);
services.Configure<KestrelServerOptions>(o => o.Limits.MaxRequestBodySize = maxBodySize);
services.Configure<FormOptions>(o =>
{
    o.ValueLengthLimit = maxBodySize;
    o.MultipartHeadersLengthLimit = maxBodySize;
});

services.AddScoped<AdministratorService>();
services.AddScoped<CacheService>();
services.AddScoped<ChatService>();
services.AddScoped<ClaimService>();
services.AddScoped<CustomerService>();
services.AddScoped<DashboardService>();
services.AddScoped<DocumentService>();
services.AddScoped<EmailService>();
services.AddScoped<InvestigatorService>();
services.AddScoped<JobService>();
services.AddScoped<LogService>();
services.AddScoped<MasterDataService>();
services.AddScoped<SearchService>();
services.AddScoped<SecurityService>();
services.AddScoped<StatusService>();

services.AddHostedService<WorkerService>();

services.AddHttpContextAccessor();
services.TryAddSingleton<IActionContextAccessor, ActionContextAccessor>();

services.AddSignalR()
    .AddNewtonsoftJsonProtocol();

services.AddSwaggerGen(o =>
{
    //o.SwaggerDoc("v1", new OpenApiInfo { Title = "Reclaim API Documentation", Description = "Welcome to the home of the Reclaim API! The Swagger doc below shows all publicly-accessible API endpoints." });
    var xmlFile = $"{Assembly.GetExecutingAssembly().GetName().Name}.xml";
    var xmlPath = Path.Combine(AppContext.BaseDirectory, xmlFile);

    if (File.Exists(xmlPath))   
        o.IncludeXmlComments(xmlPath, includeControllerXmlComments: true);

    // o.EnableAnnotations();
    // o.OperationFilter<AddUrlFormDataParamsFilter>();
    o.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
    {
        Description = @"The Reclaim API uses an Authorization header with an access token using the Bearer scheme.  Please enter your JTW access token below.",
        Name = "Authorization",
        In = ParameterLocation.Header,
        Type = SecuritySchemeType.Http,
        Scheme = "Bearer"
    });
    o.AddSecurityRequirement(new OpenApiSecurityRequirement
    {
        {
            new OpenApiSecurityScheme
            {
                Reference = new OpenApiReference
                {
                    Type = ReferenceType.SecurityScheme,
                    Id = "Bearer"
                }
            },
            new string[]{}
        }
    });
    o.OrderActionsBy(x => $"{x.ActionDescriptor.RouteValues["controller"]}_{x.ActionDescriptor}"); 
});

services.AddOpenApiDocument(document =>
{
    document.Title = "Reclaim API Documentation";
    document.Description = "Welcome to the home of the Reclaim API! The Swagger doc below shows all publicly-accessible API endpoints.";
    document.OperationProcessors.Add(new SwaggerRoleAttributeFilter());
});

services.AddRateLimiter(options =>
{
    var seconds = 60;

    options.RejectionStatusCode = 429;
    options.OnRejected = (context, _) =>
    {
        throw new ApiException(Reclaim.Api.Model.ErrorCode.RateLimitExceeded, $"Rate limit exceeded for {context.HttpContext.Request.Headers.Host.ToString()}, please try again in {seconds}s.");
    };

    if (Setting.EnforceRequestThrottling)
    {
        options.AddPolicy("authentication", context =>
        {
            var id = context.Request.Headers.Host.ToString();

            if (id.StartsWith("localhost") || id.StartsWith("10.211.55.3"))  // can't find a better way to exclude localhost from rate limiting
                id = Guid.NewGuid().ToString();

            return RateLimitPartition.GetFixedWindowLimiter(
                id,
                _ => new FixedWindowRateLimiterOptions
                {
                    AutoReplenishment = true,
                    PermitLimit = 20,
                    QueueLimit = 0,
                    Window = TimeSpan.FromSeconds(seconds),
                });
        });
    }
});

var app = builder.Build();

if (!app.Environment.IsDevelopment())
    app.UseHttpsRedirection();

app.UseAuthorization();
app.MapControllers();
app.MapHub<Reclaim.Api.Hubs.JobHub>("/hub/job", (options) =>
{
    options.Transports = Microsoft.AspNetCore.Http.Connections.HttpTransportType.WebSockets;
});

app.UseCors("Cors-Default");
app.UseMiddleware<ReplaceBearerTokenWithCookie>();
app.UseOpenApi();
app.UseSwaggerAuthorized();
app.UseSwagger();
app.UseSwaggerUI(o =>
{
    var responseFn = "(response) => { " +
        "if (response.text) { " +
        "try { var token = JSON.parse(response.text).accessToken;" +
        "   if (token) { window.localStorage.setItem('access-token', token); } } " +
        "catch (err) {}  }  " +
        "return response; }";

    var requestFn = "(request) => { " +
        "var token = window.localStorage.getItem('access-token'); " +
        "if (token && token !== 'undefined') { request.headers.Authorization = 'Bearer ' + token;}" +
        "return request; }";

    //o.EnableFilter(); 
    o.SwaggerEndpoint("/swagger/v1/swagger.json", "Reclaim API Documentation 222");
    o.InjectStylesheet("/content/swagger/custom.css");
    o.InjectJavascript("/content/swagger/custom.js");
    o.DocumentTitle = "Reclaim API Documentation";
    o.EnableTryItOutByDefault();
    o.RoutePrefix = string.Empty;
    o.UseRequestInterceptor(requestFn);
    o.UseResponseInterceptor(responseFn);    
});

app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(Path.Combine(Directory.GetCurrentDirectory(), "content")),
    RequestPath = "/content"
});

app.UseMiddleware<ApiExceptionHandler>();
app.UseRateLimiter();

using (var db = new DatabaseContext(Reclaim.Api.DbContextOptions.Get()))
{
    foreach (var setting in db.ApplicationSettings.Where(x => x.IsSecret == null))
    {
        var value = TwoWayEncryption.Encrypt(setting.Value);
        setting.IsSecret = true;
        setting.Value = value;
    }

    db.SaveChanges();
}

app.Run();


