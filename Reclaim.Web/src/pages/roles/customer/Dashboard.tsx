import { useEffect, useMemo, useState } from 'react'
import * as Model from 'api/model'
import AggregateBox from 'components/AggregateBox'
import Map from 'components/Map'
import News from 'components/News'
import StackedBarChart from 'components/StackedBarChart'
import { Log } from 'helpers/log'

const CustomerDashboard = () => {
  const dashboardApi = useMemo(
    () => new Model.DashboardClient(process.env.REACT_APP_API_URL),
    []
  )

  const [dashboard, setDashboard] = useState<Model.CustomerDashboard>()

  useEffect(() => {
    ;(async () => {
      try {
        const result = await dashboardApi.getCustomer()
        setDashboard(result)
      } catch (error) {
        Log.add(JSON.stringify(error))
      }
    })()
  }, [dashboardApi])

  return (
    <div className="dashboard">
      <div className="header">Dashboard</div>
      <div className={dashboard === undefined ? 'element-loading' : 'element-loaded'}>
        <div className="row no-gutter aggregates">
          <AggregateBox title={'Lifetime savings'} data={dashboard?.lifetimeSavings} />
          <AggregateBox title={'Recovery rate'} data={dashboard?.recoveryRate} />
          <AggregateBox
            title={'Claims under investigation'}
            data={dashboard?.claimsValueUnderInvestigation}
          />
          <AggregateBox title={'New orders'} data={dashboard?.newOrders} />
        </div>
        <div className="dashboard-chart">
          <span className="header">Monthly Investigations</span>
          <StackedBarChart data={dashboard?.claimsByMonth} />
        </div>
        <div className="row no-gutter">
          <div className="col-lg-6">
            <div className="dashboard-news">
              <span className="header">News</span>
              <News />
            </div>
          </div>
          <div className="col-lg-6">
            <div className="dashboard-map">
              <span className="header">Investigations by state</span>
              <Map data={dashboard?.claimsByState} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default CustomerDashboard
