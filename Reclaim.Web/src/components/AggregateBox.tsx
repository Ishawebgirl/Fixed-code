import * as Model from 'api/model'
import Icon from './Icon'
import HTMLReactParser from 'html-react-parser'

interface IAggregateBox {
  title: string
  data: Model.DashboardAggregate | undefined
}

const numberTriplets = /\B(?=(\d{3})+(?!\d))/g

const AggregateBox = ({ title, data }: IAggregateBox) => {
  const formatNumber = (value: number, type: string) => {
    if (type === 'Money') {
      return `$${value.toFixed(2).replace(numberTriplets, ',')}`
    } else if (type === 'Integer') {
      return `${value.toFixed(0).replace(numberTriplets, ',')}`
    } else if (type === 'Percentage') {
      return `${value.toFixed(1).replace(numberTriplets, ',')}%`
    } else {
      return value?.toString() ?? ''
    }
  }

  return data === undefined ? null : (
    <div className="aggregate col-lg-3 col-md-6">
      <div>
        <span className="header">{title}</span>
        <p>{formatNumber(data?.currentValue, data.valueType)}</p>
        <p>
          {data.percentChange >= 0 ? (
            <Icon name="ArrowTrendUp" />
          ) : (
            <Icon name="ArrowTrendDown" />
          )}
          <>
            <span className={(data?.percentChange ?? 0) >= 0 ? 'trendUp' : 'trendDown'}>
              {HTMLReactParser(
                data?.percentChange?.toFixed(1).toString() ??
                  '<span className="infinity">&infin;</span>'
              )}
              %
            </span>
            {' over previous '}
            {data?.comparisonPeriod?.toLowerCase()}
          </>
        </p>
      </div>
    </div>
  )
}
export default AggregateBox
