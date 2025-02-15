import Icon from './Icon'

interface IExportTableMenuItem {
  onClick: () => void
}

const ExportTableMenuItem = ({ onClick }: IExportTableMenuItem) => {
  return (
    <>
      <div onClick={onClick}>
        <Icon name="Download"></Icon>Export
      </div>
    </>
  )
}

export default ExportTableMenuItem
