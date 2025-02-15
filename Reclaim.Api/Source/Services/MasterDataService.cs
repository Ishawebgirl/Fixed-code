using Microsoft.EntityFrameworkCore;
using Reclaim.Api.Model;

namespace Reclaim.Api.Services;

public class MasterDataService
{
    private readonly DatabaseContext _db;
    private readonly CacheService _cacheService;

    public MasterDataService(DatabaseContext db, CacheService cacheService, DatabaseContext databaseContext)
    {
        _db = db;
        _cacheService = cacheService;
    }

    public async Task<Dictionary<EmailTemplateCode, EmailTemplate>> GetEmailTemplates()
    {
        return await _cacheService.Get(() =>
        {
            return _db.EmailTemplates
                .AsNoTracking()
                .ToDictionary(x => (EmailTemplateCode)x.ID);
        });
    }

    public async Task<EmailTemplate> GetEmailTemplate(EmailTemplateCode emailTemplateCode)
    {
        var templates = await GetEmailTemplates();

        if (templates.ContainsKey(emailTemplateCode))
            return templates[emailTemplateCode];
        else
            return null;
    }

    public async Task<Dictionary<int, State>> GetStates()
    {
        return await _cacheService.Get(() =>
        {
            return _db.States
                .AsNoTracking()
                .ToDictionary(x => x.ID);
        });
    }

    public async Task<State> GetState(int id)
    {
        var states = await GetStates();

        if (states.ContainsKey(id))
            return states[id];
        else
            return null;
    }

    public async Task<Dictionary<string, State>> GetStatesByAbbreviation()
    {
        return await _cacheService.Get(() =>
        {
            return _db.States
                .AsNoTracking()
                .ToDictionary(x => x.Code);
        });
    }

    public async Task<State> GetStateByAbbreviation(string abbreviation)
    {
        var states = await GetStatesByAbbreviation();

        if (abbreviation != null && states.ContainsKey(abbreviation))
            return states[abbreviation];
        else
            return null;
    }
}