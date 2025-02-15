import { useEffect, useMemo, useState } from 'react'
import * as Model from 'api/model'
import { useParams } from 'react-router-dom'
import { Log } from 'helpers/log'
import Menu from 'components/Menu'
import MenuItem from 'components/MenuItem'
import MenuItemDropdown from 'components/MenuItemDropdown'
import PropertyTag from 'components/PropertyTag'
import Avatar from 'components/Avatar'
import Icon from 'components/Icon'

const Claim = () => {
  const { uniqueID } = useParams()

  const claimApi = useMemo(() => new Model.ClaimClient(process.env.REACT_APP_API_URL), [])

  const [claim, setClaim] = useState<Model.Claim>()

  useEffect(() => {
    if (uniqueID === undefined) {
      return
    }

    ;(async () => {
      try {
        const result = await claimApi.getClaim(uniqueID)
        setClaim(result)
      } catch (error) {
        Log.add(JSON.stringify(error))
      }
    })()
  }, [claimApi, uniqueID])

  return (
    claim && (
      <>
        <div className="header">Claim {claim.externalID}</div>
        <Menu>
          <MenuItem label="Return to claims list" icon="ArrowLeft" linkTo=".." />
          <MenuItem label="View documents" icon="FileLines" linkTo="documents" />
          <MenuItem
            label="Chat with claim"
            icon="WandMagicSparkles"
            linkTo={`/claims/${uniqueID}/chats`}
          />
          <MenuItemDropdown
            label="Set status"
            icon="Cogs"
            typeName="ClaimStatus"
            type={Model.ClaimStatus}
          />
          <MenuItemDropdown
            label="Set disposition"
            icon="Gavel"
            typeName="ClaimDisposition"
            type={Model.ClaimDisposition}
          />
        </Menu>
        <div className="no-table">
          <div className="row">
            <div className="col-md-6 info-box">
              <div>
                <span className="header">Claim detail</span>
                <div className="cols-2">
                  {
                    // prettier-ignore
                    [
                        ['Type', <PropertyTag name={claim.type} />],
                        ['Status', <PropertyTag name={claim.status} />],
                        ['Disposition', <PropertyTag name={claim.disposition} />],
                        ['Event date', claim.eventDate.format('MM/DD/YYYY')],
                        ['Ingested timestamp', claim.ingestedTimestamp?.format('MM/DD/YYYY hh:mmA')],
                        ['Amount submitted', `$${claim.amountSubmitted}`],
                        ['Amount paid', `$${claim.amountPaid}`],
                        ['Amount adjudicated', `$${claim.amountAdjusted}`],
                      ].map(([label, value], index) => (
                        <p key={index}>
                          <span>{label}</span>
                          <div>{value}</div>
                        </p>
                      ))
                  }
                </div>
              </div>
            </div>
            <div className="col-md-6 info-box">
              <div>
                <span className="header">Policy detail</span>
                <div className="cols-2">
                  {
                    // prettier-ignore
                    [
                        ['ID', claim.policy.externalID],
                        ['Policyholder', `${claim.policy.firstName} ${claim.policy.lastName}`],
                        ['Address', <>{claim.policy.address} {claim.policy.address2}<br />{claim.policy.city}, {claim.policy.state} {claim.policy.postalCode}</>],
                        ['Carrier', claim.policy.customer.name],
                        ['Annual premium', claim.policy.annualPremium ? `$${claim.policy.annualPremium}` : 'Unknown'],
                        ['Property type', <PropertyTag name={claim.policy.propertyType} />],
                        ['Ownership type', <PropertyTag name={claim.policy.ownershipType} />],
                        ['Residence detail', `${claim.policy.bedrooms} BR / ${claim.policy.bathrooms ?? 'unknown'} BA`],
                      ].map(([label, value], index) => (
                        <p key={index}>
                        <span>{label}</span>
                        <div>{value}</div>
                        </p>
                      ))
                  }
                </div>
              </div>
            </div>
          </div>
          {claim.investigator && (
            <div className="row">
              <div className="col-md-6 info-box">
                <div>
                  <span className="header">Investigator detail</span>
                  <div>
                    <div className="half padding-right">
                      <Avatar
                        className="extra-large"
                        url={claim.investigator.avatarUrl}
                        name={`${claim.investigator.firstName} ${claim.investigator.lastName}`}
                      />
                    </div>
                    <div className="half">
                      {
                        // prettier-ignore
                        [
                          ['Name', `${claim.investigator.firstName} ${claim.investigator.lastName}`],
                          ['Address', <>{claim.investigator.address} {claim.investigator.address2}<br />{claim.investigator.city}, {claim.investigator.state} {claim.investigator.postalCode}</>],
                          ['Email address', <>{claim.investigator.emailAddress }</>],
                          ['Telephone', claim.investigator.telephone],
                          ['Status', <PropertyTag name={claim.investigator.status} />],
                      ].map(([label, value], index) => (
                        <p key={index}>
                          <span>{label}</span>
                          <div>{value}</div>
                        </p>
                      ))
                      }
                    </div>
                  </div>
                </div>
              </div>
              <div className="col-md-6 info-box">
                <div>
                  <span className="header">Status</span>
                  <div className="statuses">
                    {
                      // prettier-ignore
                      [
                          ['Bell', 'Event occurred', `${claim.eventDate.format('MM/DD/yyyy')} - ${claim.policy.firstName} ${claim.policy.lastName}`],
                          ['Upload', 'Claim submitted', `${claim.eventDate.add('d', 7).format('MM/DD/yyyy')} - ${claim.policy.customer.name}`],
                          ['Gears', 'Claim ingested', `${claim.ingestedTimestamp?.format('MM/DD/yyyy')}`],
                          claim.investigator && ['Clipboard', 'Investigation started', `${claim.eventDate.add('d', 50).format('MM/DD/yyyy')} - ${claim.investigator?.firstName} ${claim.investigator?.lastName}`],
                          claim.investigator && claim.adjudicatedTimestamp && ['Gavel', 'Investigation completed', `${claim.adjudicatedTimestamp?.format('MM/DD/yyyy')} - ${claim.investigator?.firstName} ${claim.investigator?.lastName}`]
                          ].filter((item): item is [string, string, string] => Boolean(item)).map(([icon, label, details], index) => (
                        <div key={index}>
                          <Icon className="property-tag" name={icon} />
                          <div className="details">
                          {label}
                          <br></br>
                          <span>{details}</span>
                          </div>
                        </div>
                        ))
                    }
                  </div>
                </div>
              </div>
            </div>
          )}{' '}
        </div>
      </>
    )
  )
}

export default Claim
